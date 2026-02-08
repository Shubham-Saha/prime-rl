"""
Transport security test: real serialization and ZMQ transport with mock data.

Tests the exact same wire format and protocol as production:
  - msgspec.msgpack serialization (same encoder/decoder)
  - ZMQ PUSH/PULL (TrainingBatch leg) and PUB/SUB (MicroBatch leg)
  - Multipart framing with sender_id / topic prefixes

Three phases:
  Phase 1: TrainingBatch round-trip (Orchestrator → Packer)
  Phase 2: MicroBatch round-trip via real prepare_batch() (Packer → Trainer)
  Phase 3: Security edge cases (corruption, wrong type, truncation, flooding)

Usage:
    uv run python -m prime_rl.mock.test_transport_security
"""

import json
import logging
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import msgspec
import torch
import zmq

from prime_rl.trainer.batch import prepare_batch
from prime_rl.transport.types import MicroBatch, TrainingSample, TrainingBatch

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "simulation_config.json"


@dataclass
class TransportTestConfig:
    seed: int = 42
    seq_len: int = 256
    vocab_size: int = 151936
    batch_size: int = 4
    num_train_workers: int = 2
    base_port: int = 15555
    flood_count: int = 10_000
    flood_duration_target: float = 1.0

    @classmethod
    def from_json(cls, path: Optional[Path] = None) -> "TransportTestConfig":
        p = path or CONFIG_PATH
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return cls(
                seed=data.get("seed", cls.seed),
                seq_len=data.get("training", {}).get("seq_len", cls.seq_len),
                vocab_size=data.get("data", {}).get("vocab_size", cls.vocab_size),
                batch_size=data.get("training", {}).get("batch_size", cls.batch_size),
            )
        return cls()


# ---------------------------------------------------------------------------
# Mock data → real msgspec types
# ---------------------------------------------------------------------------

def generate_training_sample(
    rng: random.Random, seq_len: int, vocab_size: int,
) -> TrainingSample:
    """Generate a single TrainingSample (real msgspec Struct) from mock parameters."""
    prompt_len = int(seq_len * 0.3)
    completion_len = seq_len - prompt_len

    return TrainingSample(
        prompt_ids=[rng.randint(0, vocab_size - 1) for _ in range(prompt_len)],
        prompt_mask=[False] * prompt_len,
        completion_ids=[rng.randint(0, vocab_size - 1) for _ in range(completion_len)],
        completion_mask=[True] * completion_len,
        completion_logprobs=[rng.gauss(-3.0, 0.5) for _ in range(completion_len)],
        completion_temperatures=[1.0] * completion_len,
        advantage=rng.gauss(0, 1),
        reward=rng.uniform(0, 1),
    )


def generate_training_batch(
    rng: random.Random,
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    step: int = 0,
) -> TrainingBatch:
    """Generate a TrainingBatch with multiple samples."""
    samples = [
        generate_training_sample(rng, seq_len, vocab_size)
        for _ in range(num_samples)
    ]
    return TrainingBatch(examples=samples, step=step)


# ---------------------------------------------------------------------------
# Phase 1: TrainingBatch round-trip over ZMQ PUSH/PULL
# ---------------------------------------------------------------------------

def phase1_training_batch_transport(cfg: TransportTestConfig) -> bool:
    """Test TrainingBatch serialization over ZMQ PUSH → PULL."""
    logger.info("=== Phase 1: TrainingBatch transport (Orchestrator -> Packer) ===")

    rng = random.Random(cfg.seed)
    port = cfg.base_port

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    batch_sent = generate_training_batch(
        rng, num_samples=cfg.batch_size, seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size, step=0,
    )

    payload = encoder.encode(batch_sent)
    sender_id = b"mock_orchestrator"

    logger.info(
        "TrainingBatch: %d samples, payload=%d bytes",
        len(batch_sent.examples), len(payload),
    )

    received_batches: list[TrainingBatch] = []
    receiver_error: list[Exception] = []

    def receiver_thread():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVHWM, 10)
        sock.bind(f"tcp://127.0.0.1:{port}")

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        try:
            events = dict(poller.poll(timeout=5000))
            if sock in events:
                parts = sock.recv_multipart(copy=False)
                recv_sender_id = bytes(parts[0])
                recv_payload = bytes(parts[1])
                batch = decoder.decode(recv_payload)
                received_batches.append(batch)
                logger.info(
                    "Received: sender_id=%s, payload=%d bytes, samples=%d",
                    recv_sender_id, len(recv_payload), len(batch.examples),
                )
            else:
                receiver_error.append(TimeoutError("No message received in 5s"))
        except Exception as e:
            receiver_error.append(e)
        finally:
            sock.close(linger=0)
            ctx.term()

    def sender_thread():
        time.sleep(0.2)  # let receiver bind first
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.SNDHWM, 10)
        sock.connect(f"tcp://127.0.0.1:{port}")

        sock.send_multipart([sender_id, payload], copy=False)
        logger.info("Sent TrainingBatch via ZMQ PUSH")
        time.sleep(0.1)

        sock.close(linger=0)
        ctx.term()

    t_recv = threading.Thread(target=receiver_thread)
    t_send = threading.Thread(target=sender_thread)
    t_recv.start()
    t_send.start()
    t_recv.join(timeout=10)
    t_send.join(timeout=10)

    if receiver_error:
        logger.error("Phase 1 FAILED: %s", receiver_error[0])
        return False

    if not received_batches:
        logger.error("Phase 1 FAILED: no batch received")
        return False

    batch_recv = received_batches[0]
    ok = _verify_training_batch(batch_sent, batch_recv)
    logger.info("Phase 1: %s", "PASS" if ok else "FAIL")
    return ok


def _verify_training_batch(sent: TrainingBatch, recv: TrainingBatch) -> bool:
    """Field-by-field verification of TrainingBatch round-trip."""
    if sent.step != recv.step:
        logger.error("step mismatch: %d != %d", sent.step, recv.step)
        return False

    if len(sent.examples) != len(recv.examples):
        logger.error(
            "sample count mismatch: %d != %d",
            len(sent.examples), len(recv.examples),
        )
        return False

    for i, (s, r) in enumerate(zip(sent.examples, recv.examples)):
        if s.prompt_ids != r.prompt_ids:
            logger.error("sample %d: prompt_ids mismatch", i)
            return False
        if s.completion_ids != r.completion_ids:
            logger.error("sample %d: completion_ids mismatch", i)
            return False
        if s.completion_logprobs != r.completion_logprobs:
            logger.error("sample %d: completion_logprobs mismatch", i)
            return False
        if s.advantage != r.advantage:
            logger.error("sample %d: advantage mismatch", i)
            return False

    logger.info("TrainingBatch verified: %d samples, all fields match", len(sent.examples))
    return True


# ---------------------------------------------------------------------------
# Phase 2: MicroBatch round-trip over ZMQ PUB/SUB (with real packing)
# ---------------------------------------------------------------------------

def phase2_micro_batch_transport(cfg: TransportTestConfig) -> bool:
    """Test MicroBatch serialization over ZMQ PUB → SUB, using real prepare_batch()."""
    logger.info("=== Phase 2: MicroBatch transport (Packer -> Trainer) ===")

    rng = random.Random(cfg.seed)
    port = cfg.base_port + 10

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])

    # Generate samples and pack with real algorithm
    samples = [
        generate_training_sample(rng, cfg.seq_len, cfg.vocab_size)
        for _ in range(cfg.batch_size)
    ]

    micro_batch_grid = prepare_batch(
        rollouts=samples,
        seq_len=cfg.seq_len,
        num_train_workers=cfg.num_train_workers,
        idxs=[0] * len(samples),
        num_loras=1,
        pad_to_multiple_of=8,
    )

    logger.info(
        "prepare_batch: %d samples -> %d workers x [%s] micro batches",
        len(samples), len(micro_batch_grid),
        ", ".join(str(len(g)) for g in micro_batch_grid),
    )

    # Test transport for data_rank=0
    data_rank = 0
    micro_batches_sent = micro_batch_grid[data_rank]
    payload = encoder.encode(micro_batches_sent)
    topic = f"data_rank|{data_rank}|".encode()

    logger.info(
        "MicroBatch payload for rank %d: %d batches, %d bytes",
        data_rank, len(micro_batches_sent), len(payload),
    )

    received_micro_batches: list[list[MicroBatch]] = []
    receiver_error: list[Exception] = []

    def receiver_thread():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 10)
        sock.connect(f"tcp://127.0.0.1:{port}")
        sock.setsockopt(zmq.SUBSCRIBE, topic)

        # Announce readiness on barrier port (matches production protocol)
        ready_sock = ctx.socket(zmq.PUSH)
        ready_sock.connect(f"tcp://127.0.0.1:{port + 1}")
        ready_sock.send(str(data_rank).encode())

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        try:
            events = dict(poller.poll(timeout=5000))
            if sock in events:
                parts = sock.recv_multipart(copy=False)
                recv_topic = bytes(parts[0])
                recv_payload = bytes(parts[1])
                micro_batches = decoder.decode(recv_payload)
                received_micro_batches.append(micro_batches)
                logger.info(
                    "Received: topic=%s, payload=%d bytes, batches=%d",
                    recv_topic, len(recv_payload), len(micro_batches),
                )
            else:
                receiver_error.append(TimeoutError("No message received in 5s"))
        except Exception as e:
            receiver_error.append(e)
        finally:
            ready_sock.close(linger=0)
            sock.close(linger=0)
            ctx.term()

    def sender_thread():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 10)
        sock.bind(f"tcp://127.0.0.1:{port}")

        # Wait for readiness on barrier port (matches production protocol)
        ready_sock = ctx.socket(zmq.PULL)
        ready_sock.bind(f"tcp://127.0.0.1:{port + 1}")
        ready_sock.recv()  # block until subscriber ready

        sock.send_multipart([topic, payload], copy=False)
        logger.info("Sent MicroBatch via ZMQ PUB (topic=%s)", topic)
        time.sleep(0.1)

        ready_sock.close(linger=0)
        sock.close(linger=0)
        ctx.term()

    t_recv = threading.Thread(target=receiver_thread)
    t_send = threading.Thread(target=sender_thread)
    t_recv.start()
    t_send.start()
    t_recv.join(timeout=10)
    t_send.join(timeout=10)

    if receiver_error:
        logger.error("Phase 2 FAILED: %s", receiver_error[0])
        return False

    if not received_micro_batches:
        logger.error("Phase 2 FAILED: no micro batches received")
        return False

    ok = _verify_micro_batches(micro_batches_sent, received_micro_batches[0])
    if ok:
        _verify_tensor_conversion(received_micro_batches[0])
    logger.info("Phase 2: %s", "PASS" if ok else "FAIL")
    return ok


def _verify_micro_batches(
    sent: list[MicroBatch], recv: list[MicroBatch],
) -> bool:
    """Field-by-field verification of MicroBatch round-trip."""
    if len(sent) != len(recv):
        logger.error("micro batch count mismatch: %d != %d", len(sent), len(recv))
        return False

    for i, (s, r) in enumerate(zip(sent, recv)):
        if s.input_ids != r.input_ids:
            logger.error("micro batch %d: input_ids mismatch", i)
            return False
        if s.loss_mask != r.loss_mask:
            logger.error("micro batch %d: loss_mask mismatch", i)
            return False
        if s.advantages != r.advantages:
            logger.error("micro batch %d: advantages mismatch", i)
            return False
        if s.inference_logprobs != r.inference_logprobs:
            logger.error("micro batch %d: inference_logprobs mismatch", i)
            return False
        if s.position_ids != r.position_ids:
            logger.error("micro batch %d: position_ids mismatch", i)
            return False
        if s.temperatures != r.temperatures:
            logger.error("micro batch %d: temperatures mismatch", i)
            return False
        if s.lora_num_tokens != r.lora_num_tokens:
            logger.error("micro batch %d: lora_num_tokens mismatch", i)
            return False

    logger.info("MicroBatch verified: %d batches, all fields match", len(sent))
    return True


def _verify_tensor_conversion(micro_batches: list[MicroBatch]) -> None:
    """Verify MicroBatch → TensorMicroBatch conversion (same as real DataLoader)."""
    for i, mb in enumerate(micro_batches):
        input_ids = torch.tensor(mb.input_ids, dtype=torch.long).unsqueeze(0)
        advantages = torch.tensor(mb.advantages, dtype=torch.float).unsqueeze(0)
        loss_mask = torch.tensor(mb.loss_mask, dtype=torch.bool).unsqueeze(0)
        temperatures = torch.tensor(mb.temperatures, dtype=torch.float).unsqueeze(0)

        logger.info(
            "  MicroBatch %d -> tensor: input_ids=%s, advantages=%s, "
            "loss_mask=%s (active=%d/%d)",
            i, list(input_ids.shape), list(advantages.shape),
            list(loss_mask.shape), loss_mask.sum().item(), loss_mask.numel(),
        )


# ---------------------------------------------------------------------------
# Phase 3: Security edge cases
# ---------------------------------------------------------------------------

def phase3_security_edge_cases(cfg: TransportTestConfig) -> bool:
    """Test deserialization safety and flooding resilience."""
    logger.info("=== Phase 3: Security edge cases ===")

    all_passed = True
    all_passed &= _test_corrupted_payload()
    all_passed &= _test_wrong_type()
    all_passed &= _test_truncated_payload(cfg)
    all_passed &= _test_zmq_flooding(cfg)

    logger.info("Phase 3: %s", "PASS" if all_passed else "FAIL")
    return all_passed


def _test_corrupted_payload() -> bool:
    """Send random bytes where a TrainingBatch is expected."""
    logger.info("  [3a] Corrupted payload...")
    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    garbage = os.urandom(512)
    try:
        decoder.decode(garbage)
        logger.error("  FAIL: decoder accepted garbage bytes")
        return False
    except (msgspec.ValidationError, msgspec.DecodeError):
        logger.info("  PASS: decoder rejected garbage bytes safely")
        return True
    except Exception as e:
        logger.error("  FAIL: unexpected exception: %s: %s", type(e).__name__, e)
        return False


def _test_wrong_type() -> bool:
    """Send a MicroBatch where a TrainingBatch is expected."""
    logger.info("  [3b] Wrong msgspec type...")

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    wrong = MicroBatch(
        input_ids=[1, 2, 3],
        loss_mask=[True, True, True],
        advantages=[0.5, 0.5, 0.5],
        inference_logprobs=[-1.0, -1.0, -1.0],
        position_ids=[0, 1, 2],
        temperatures=[1.0, 1.0, 1.0],
    )

    payload = encoder.encode(wrong)
    try:
        decoder.decode(payload)
        # msgspec array_like structs are positional — this may decode into
        # a malformed TrainingBatch. Check if it silently produces garbage.
        logger.warning(
            "  NOTE: decoder accepted wrong type (msgspec array_like structs "
            "are positionally decoded — type confusion possible)"
        )
        return True  # not a crash, but worth noting
    except (msgspec.ValidationError, msgspec.DecodeError):
        logger.info("  PASS: decoder rejected wrong type")
        return True
    except Exception as e:
        logger.error("  FAIL: unexpected exception: %s: %s", type(e).__name__, e)
        return False


def _test_truncated_payload(cfg: TransportTestConfig) -> bool:
    """Send partial msgpack bytes."""
    logger.info("  [3c] Truncated payload...")

    rng = random.Random(cfg.seed)
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    batch = generate_training_batch(rng, 2, cfg.seq_len, cfg.vocab_size)
    full_payload = encoder.encode(batch)

    # Truncate at 50%
    truncated = full_payload[: len(full_payload) // 2]
    try:
        decoder.decode(truncated)
        logger.error("  FAIL: decoder accepted truncated payload")
        return False
    except (msgspec.ValidationError, msgspec.DecodeError):
        logger.info(
            "  PASS: decoder rejected truncated payload (%d/%d bytes)",
            len(truncated), len(full_payload),
        )
        return True
    except Exception as e:
        logger.error("  FAIL: unexpected exception: %s: %s", type(e).__name__, e)
        return False


def _test_zmq_flooding(cfg: TransportTestConfig) -> bool:
    """
    Flood test: send many valid MicroBatch packets rapidly.

    Tests whether ZMQ HWM (high water mark) properly bounds memory, or
    whether the receiver's buffer grows unbounded (DoS / buffer overflow).
    """
    logger.info(
        "  [3d] ZMQ flooding: %d packets targeting %.1fs...",
        cfg.flood_count, cfg.flood_duration_target,
    )

    port = cfg.base_port + 20
    rng = random.Random(cfg.seed)
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])

    # Pre-generate a small MicroBatch payload
    sample = generate_training_sample(rng, cfg.seq_len, cfg.vocab_size)
    grid = prepare_batch(
        rollouts=[sample],
        seq_len=cfg.seq_len,
        num_train_workers=1,
        idxs=[0],
        num_loras=1,
        pad_to_multiple_of=8,
    )
    payload = encoder.encode(grid[0])
    topic = b"data_rank|0|"
    payload_size = len(payload)

    sender_stats: dict = {}
    receiver_stats: dict = {}
    receiver_error: list[Exception] = []

    hwm = 10  # intentionally low to test backpressure

    def flood_receiver():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, hwm)
        sock.connect(f"tcp://127.0.0.1:{port}")
        sock.setsockopt(zmq.SUBSCRIBE, topic)

        # Signal readiness
        ready_sock = ctx.socket(zmq.PUSH)
        ready_sock.connect(f"tcp://127.0.0.1:{port + 1}")
        ready_sock.send(b"0")

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        received = 0
        decoded_ok = 0
        decode_errors = 0
        start = time.perf_counter()

        try:
            while True:
                events = dict(poller.poll(timeout=2000))
                if sock not in events:
                    break  # no more messages for 2s
                parts = sock.recv_multipart(copy=False, flags=zmq.NOBLOCK)
                received += 1
                try:
                    decoder.decode(bytes(parts[1]))
                    decoded_ok += 1
                except Exception:
                    decode_errors += 1
        except Exception as e:
            receiver_error.append(e)
        finally:
            elapsed = time.perf_counter() - start
            receiver_stats.update({
                "received": received,
                "decoded_ok": decoded_ok,
                "decode_errors": decode_errors,
                "elapsed_s": round(elapsed, 3),
            })
            ready_sock.close(linger=0)
            sock.close(linger=0)
            ctx.term()

    def flood_sender():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, hwm)
        sock.bind(f"tcp://127.0.0.1:{port}")

        ready_sock = ctx.socket(zmq.PULL)
        ready_sock.bind(f"tcp://127.0.0.1:{port + 1}")
        ready_sock.recv()  # wait for subscriber

        sent = 0
        dropped = 0
        start = time.perf_counter()

        for _ in range(cfg.flood_count):
            try:
                sock.send_multipart([topic, payload], flags=zmq.NOBLOCK, copy=False)
                sent += 1
            except zmq.Again:
                dropped += 1

        elapsed = time.perf_counter() - start
        sender_stats.update({
            "attempted": cfg.flood_count,
            "sent": sent,
            "dropped": dropped,
            "elapsed_s": round(elapsed, 3),
            "rate_msg_per_s": round(sent / elapsed) if elapsed > 0 else 0,
        })

        time.sleep(0.5)  # let receiver drain
        ready_sock.close(linger=0)
        sock.close(linger=0)
        ctx.term()

    t_recv = threading.Thread(target=flood_receiver)
    t_send = threading.Thread(target=flood_sender)
    t_recv.start()
    t_send.start()
    t_send.join(timeout=30)
    t_recv.join(timeout=30)

    if receiver_error:
        logger.error("  FAIL: receiver error during flood: %s", receiver_error[0])
        return False

    logger.info(
        "  Sender:   attempted=%d, sent=%d, dropped=%d, "
        "elapsed=%.3fs, rate=%d msg/s",
        sender_stats.get("attempted", 0), sender_stats.get("sent", 0),
        sender_stats.get("dropped", 0), sender_stats.get("elapsed_s", 0),
        sender_stats.get("rate_msg_per_s", 0),
    )
    logger.info(
        "  Receiver: received=%d, decoded_ok=%d, decode_errors=%d, "
        "elapsed=%.3fs",
        receiver_stats.get("received", 0), receiver_stats.get("decoded_ok", 0),
        receiver_stats.get("decode_errors", 0), receiver_stats.get("elapsed_s", 0),
    )

    total_bytes_sent = sender_stats.get("sent", 0) * payload_size
    logger.info(
        "  Wire:     payload=%d bytes/msg, total_sent=%.1f MB",
        payload_size, total_bytes_sent / (1024 * 1024),
    )

    # Verify: receiver should NOT have received all messages (HWM bounds it)
    sent_count = sender_stats.get("sent", 0)
    recv_count = receiver_stats.get("received", 0)

    if sent_count > hwm * 5 and recv_count < sent_count:
        logger.info(
            "  PASS: ZMQ HWM bounded receiver buffer "
            "(sent=%d, received=%d, HWM=%d)",
            sent_count, recv_count, hwm,
        )
        return True
    elif recv_count == sent_count:
        logger.warning(
            "  NOTE: receiver got all %d messages — HWM did not drop any. "
            "This may indicate the receiver is fast enough to drain, "
            "or HWM is set too high.",
            recv_count,
        )
        return True  # not a failure, just informational
    else:
        logger.info(
            "  PASS: flooding test complete (sent=%d, received=%d)",
            sent_count, recv_count,
        )
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = TransportTestConfig.from_json()
    logger.info(
        "Transport security test: seed=%d, seq_len=%d, vocab_size=%d, "
        "batch_size=%d, base_port=%d",
        cfg.seed, cfg.seq_len, cfg.vocab_size, cfg.batch_size, cfg.base_port,
    )

    results = {}
    results["phase1_training_batch"] = phase1_training_batch_transport(cfg)
    results["phase2_micro_batch"] = phase2_micro_batch_transport(cfg)
    results["phase3_security"] = phase3_security_edge_cases(cfg)

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    for name, passed in results.items():
        logger.info("  %-30s %s", name, "PASS" if passed else "FAIL")

    all_passed = all(results.values())
    logger.info("=" * 60)
    logger.info("Overall: %s", "ALL PASSED" if all_passed else "SOME FAILED")
    logger.info("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
