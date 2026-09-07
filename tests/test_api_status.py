import asyncio

import api_router
from api_router import check_status
from version import __version__


def test_status_reports_current_version_without_credentials(monkeypatch):
    # 模拟未配置任何 API Key 的场景，避免依赖实际 .env 环境
    monkeypatch.setattr(api_router, "SILICONFLOW_API_KEY", "")
    monkeypatch.setattr(api_router, "MAGICK_API_KEY", "")

    status = asyncio.run(check_status())

    assert status["status"] == "healthy"
    assert status["version"] == __version__
    assert status["siliconflow_configured"] is False
    assert status["magick_configured"] is False
