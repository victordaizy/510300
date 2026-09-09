"""续取仅重试传输错误，权限和参数错误不得反复请求。"""
import pytest
from research.nbs_v2_source_refetch_v1_2 import retryable


@pytest.mark.parametrize("code", ["CONNECTIONERROR", "READTIMEOUT", "TIMEOUT", "CONNECTTIMEOUT"])
def test_transport_errors_can_retry(code):
    assert retryable({"business_code": code, "http_status": None})


@pytest.mark.parametrize("code", ["40203", "401", "INVALID_PARAMETER", "INVALID_BODY"])
def test_permission_and_parameter_errors_cannot_retry(code):
    assert not retryable({"business_code": code, "http_status": 200})


@pytest.mark.parametrize("status,allowed", [(429, True), (502, True), (403, False), (401, False)])
def test_http_retry_boundary(status, allowed):
    assert retryable({"http_status": status}) == allowed
