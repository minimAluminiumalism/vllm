# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# type: ignore
import threading
from collections.abc import Iterable
from concurrent import futures
from typing import Callable, Generator, Literal

import grpc
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TRACES_INSECURE
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace

from vllm import LLM, SamplingParams
from vllm.tracing import SpanAttributes, get_trace_headers_from_current_context

FAKE_TRACE_SERVER_ADDRESS = "localhost:4317"

FieldName = Literal[
    "bool_value", "string_value", "int_value", "double_value", "array_value"
]


def decode_value(value: AnyValue):
    field_decoders: dict[FieldName, Callable] = {
        "bool_value": (lambda v: v.bool_value),
        "string_value": (lambda v: v.string_value),
        "int_value": (lambda v: v.int_value),
        "double_value": (lambda v: v.double_value),
        "array_value": (
            lambda v: [decode_value(item) for item in v.array_value.values]
        ),
    }
    for field, decoder in field_decoders.items():
        if value.HasField(field):
            return decoder(value)
    raise ValueError(f"Couldn't decode value: {value}")


def decode_attributes(attributes: Iterable[KeyValue]):
    return {kv.key: decode_value(kv.value) for kv in attributes}


class FakeTraceService(TraceServiceServicer):
    def __init__(self):
        self.request = None
        self.evt = threading.Event()

    def Export(self, request, context):
        self.request = request
        self.evt.set()
        return ExportTraceServiceResponse()


@pytest.fixture
def trace_service() -> Generator[FakeTraceService, None, None]:
    """Fixture to set up a fake gRPC trace service"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    service = FakeTraceService()
    add_TraceServiceServicer_to_server(service, server)
    server.add_insecure_port(FAKE_TRACE_SERVER_ADDRESS)
    server.start()

    yield service

    server.stop(None)


def test_traces(
    monkeypatch: pytest.MonkeyPatch,
    trace_service: FakeTraceService,
):
    with monkeypatch.context() as m:
        m.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")

        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.1,
            max_tokens=256,
        )
        model = "facebook/opt-125m"
        llm = LLM(
            model=model,
            otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
            gpu_memory_utilization=0.3,
            disable_log_stats=False,
        )
        prompts = ["This is a short prompt"]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        print(f"test_traces outputs is : {outputs}")

        timeout = 10
        if not trace_service.evt.wait(timeout):
            raise TimeoutError(
                f"The fake trace service didn't receive a trace within "
                f"the {timeout} seconds timeout"
            )

        request = trace_service.request
        assert len(request.resource_spans) == 1, (
            f"Expected 1 resource span, but got {len(request.resource_spans)}"
        )
        assert len(request.resource_spans[0].scope_spans) == 1, (
            f"Expected 1 scope span, "
            f"but got {len(request.resource_spans[0].scope_spans)}"
        )
        assert len(request.resource_spans[0].scope_spans[0].spans) == 1, (
            f"Expected 1 span, "
            f"but got {len(request.resource_spans[0].scope_spans[0].spans)}"
        )

        attributes = decode_attributes(
            request.resource_spans[0].scope_spans[0].spans[0].attributes
        )
        # assert attributes.get(SpanAttributes.GEN_AI_RESPONSE_MODEL) == model
        assert attributes.get(SpanAttributes.GEN_AI_REQUEST_ID) == outputs[0].request_id
        assert (
            attributes.get(SpanAttributes.GEN_AI_REQUEST_TEMPERATURE)
            == sampling_params.temperature
        )
        assert (
            attributes.get(SpanAttributes.GEN_AI_REQUEST_TOP_P) == sampling_params.top_p
        )
        assert (
            attributes.get(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS)
            == sampling_params.max_tokens
        )
        assert attributes.get(SpanAttributes.GEN_AI_REQUEST_N) == sampling_params.n
        assert attributes.get(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS) == len(
            outputs[0].prompt_token_ids
        )
        completion_tokens = sum(len(o.token_ids) for o in outputs[0].outputs)
        assert (
            attributes.get(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS)
            == completion_tokens
        )

        assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE) > 0
        assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN) > 0
        assert attributes.get(SpanAttributes.GEN_AI_LATENCY_E2E) > 0


def test_trace_context_propagation(
    monkeypatch: pytest.MonkeyPatch,
    trace_service: FakeTraceService,
):
    """Test that vLLM automatically detects and propagates trace context
    from an active OpenTelemetry span in offline inference scenarios.

    This test verifies issue #32177: offline inference should automatically
    detect the current OpenTelemetry context and link vLLM spans to it.
    """
    with monkeypatch.context() as m:
        m.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")

        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.1,
            max_tokens=64,
        )
        model = "facebook/opt-125m"

        # Create LLM first - this will set up vLLM's tracer provider
        llm = LLM(
            model=model,
            otlp_traces_endpoint=FAKE_TRACE_SERVER_ADDRESS,
            gpu_memory_utilization=0.3,
            disable_log_stats=False,
        )

        # Get the tracer from the global provider (which vLLM has set up)
        # This ensures we use the same provider as vLLM
        tracer = trace.get_tracer("test-tracer")

        # Create a parent span and call llm.generate() within its context
        # vLLM should automatically detect this span and link to it
        with tracer.start_as_current_span("rag-pipeline") as parent_span:
            parent_trace_id = parent_span.get_span_context().trace_id
            parent_span_id = parent_span.get_span_context().span_id

            prompts = ["Hello, world!"]
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            print(f"test_trace_context_propagation outputs: {outputs}")

        # Force flush the spans from the global provider
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()

        timeout = 15
        if not trace_service.evt.wait(timeout):
            raise TimeoutError(
                f"The fake trace service didn't receive expected spans within "
                f"the {timeout} seconds timeout"
            )

        # Get the vLLM span from the request
        request = trace_service.request
        assert len(request.resource_spans) >= 1, (
            f"Expected at least 1 resource span, but got {len(request.resource_spans)}"
        )

        # Find the vLLM span, it should have gen_ai attributes
        vllm_span = None
        for resource_span in request.resource_spans:
            for scope_span in resource_span.scope_spans:
                for span in scope_span.spans:
                    attrs = decode_attributes(span.attributes)
                    if SpanAttributes.GEN_AI_REQUEST_ID in attrs:
                        vllm_span = span
                        break

        assert vllm_span is not None, "Could not find vLLM span in received spans"

        # Verify that the vLLM span has the correct trace_id
        vllm_trace_id = int.from_bytes(vllm_span.trace_id, byteorder="big")
        assert vllm_trace_id == parent_trace_id, (
            f"vLLM span trace_id ({vllm_trace_id}) should match "
            f"parent span trace_id ({parent_trace_id})"
        )

        # Verify that the vLLM span's parent_span_id matches the parent span's span_id
        vllm_parent_span_id = int.from_bytes(vllm_span.parent_span_id, byteorder="big")
        assert vllm_parent_span_id == parent_span_id, (
            f"vLLM span parent_span_id ({vllm_parent_span_id}) should match "
            f"parent span span_id ({parent_span_id})"
        )

        # Verify the vLLM span has the expected attributes
        vllm_attrs = decode_attributes(vllm_span.attributes)
        assert vllm_attrs.get(SpanAttributes.GEN_AI_REQUEST_ID) == outputs[0].request_id


def test_get_trace_headers_from_current_context_with_active_span():
    """Test that get_trace_headers_from_current_context returns headers
    when there's an active span."""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("test-tracer")

    # Without an active span, should return None
    result = get_trace_headers_from_current_context()
    assert result is None, "Should return None when no active span"

    # With an active span, should return trace headers
    with tracer.start_as_current_span("test-span") as span:
        result = get_trace_headers_from_current_context()
        assert result is not None, "Should return headers when span is active"
        assert "traceparent" in result, "Should contain traceparent header"

        # Verify the traceparent contains the correct trace_id
        trace_id = span.get_span_context().trace_id
        span_id = span.get_span_context().span_id
        traceparent = result["traceparent"]

        # Verify traceparent header. e.g., "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        parts = traceparent.split("-")
        assert len(parts) == 4, f"traceparent should have 4 parts: {traceparent}"

        # Verify trace_id matches
        expected_trace_id = format(trace_id, "032x")
        assert parts[1] == expected_trace_id, (
            f"trace_id mismatch: {parts[1]} != {expected_trace_id}"
        )

        # Verify span_id matches
        expected_span_id = format(span_id, "016x")
        assert parts[2] == expected_span_id, (
            f"span_id mismatch: {parts[2]} != {expected_span_id}"
        )

    # After exiting the span context, should return None again
    result = get_trace_headers_from_current_context()
    assert result is None, "Should return None after span context exits"
