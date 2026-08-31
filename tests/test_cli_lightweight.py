"""Tests that the light CLI surface works without the heavy dependency stack."""

from evaloop.utils.validation import SystemValidator


def test_cli_module_imports_without_heavy_stack():
    # Importing the CLI must not require torch/evalplus (they are lazy)
    import evaloop.cli  # noqa: F401


def test_validate_all_returns_check_dicts():
    results = SystemValidator().validate_all(check_gpu=False, check_api_keys=False)
    assert set(results) == {"python_version", "required_packages", "disk_space"}
    for check in results.values():
        assert isinstance(check["passed"], bool)
        assert isinstance(check["message"], str)


def test_gpu_checks_survive_missing_torch():
    # Regardless of whether torch is installed, the GPU checks must return
    # a well-formed result instead of raising.
    validator = SystemValidator()
    gpu = validator._check_gpu_availability()
    cuda = validator._check_cuda_version()
    assert "passed" in gpu and "message" in gpu
    assert "passed" in cuda and "message" in cuda
