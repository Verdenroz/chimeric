from pathlib import Path

import vcr
from vcr.record_mode import RecordMode


def get_vcr() -> vcr.VCR:
    """Get configured VCR instance for recording/playing back HTTP interactions.

    Returns:
        Configured VCR instance.
    """
    cassette_dir = Path(__file__).parent / "cassettes"
    cassette_dir.mkdir(exist_ok=True)

    return vcr.VCR(
        # Cassette configuration
        cassette_library_dir=str(cassette_dir),
        # Request matching configuration
        match_on=["method", "scheme", "host", "port", "path", "query"],
        # Filter sensitive information
        filter_headers=[
            "authorization",
            "x-api-key",
            "api-key",
            "openai-api-key",
            "anthropic-api-key",
            "google-api-key",
            "cerebras-api-key",
            "cohere-api-key",
            "grok-api-key",
            "groq-api-key",
            "x-goog-api-key",
            "x-goog-request-id",
            "x-request-id",
            "x-amzn-requestid",
            "cf-ray",
            "anthropic-organization-id",
            "x-debug-trace-id",
            "x-trace-id",
            "x-correlation-id",
            "server",
            "date",
            "user-agent",
        ],
        # Filter query parameters that might contain sensitive info
        filter_query_parameters=[
            "key",
            "api_key",
            "apikey",
            "token",
            "access_token",
        ],
        # Recording configuration
        record_mode=RecordMode.NEW_EPISODES,  # Record new episodes if not found
        # Decode compressed responses for better cassette readability
        decode_compressed_response=True,
        # Configure serializer for better cassette format
        serializer="yaml",
        # Custom request/response filtering
        before_record_request=_filter_request,
        before_record_response=_filter_response,
    )


def _filter_request(request):
    """Filter and sanitize request data before recording."""
    # Note: Header filtering is handled by VCR's filter_headers configuration
    # This function is available for additional request body filtering if needed

    # Filter request body if it contains sensitive information
    if hasattr(request, "body") and request.body:
        pass
        # Add request body filtering here if needed in the future
        # For now, header filtering is sufficient

    return request


def _filter_response(response):
    """Filter and sanitize response data before recording."""
    # Remove any response headers that might contain sensitive info
    sensitive_response_headers = ["set-cookie", "x-request-id"]

    for header in sensitive_response_headers:
        if header in response.get("headers", {}):
            del response["headers"][header]

    return response


def get_cassette_path(provider: str, test_name: str) -> str:
    """Get the cassette file path for a specific provider and test.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        test_name: Test function name

    Returns:
        Path to the cassette file.
    """
    cassette_dir = Path(__file__).parent / "cassettes" / provider
    cassette_dir.mkdir(parents=True, exist_ok=True)
    return str(cassette_dir / f"{test_name}.yaml")
