from openquant.risk.guardrails import apply_guardrails


def test_apply_guardrails_flags_breaches():
    ok, reasons = apply_guardrails(
        max_drawdown=0.30,
        cvar=0.10,
        worst_daily_loss=0.07,
        dd_limit=0.20,
        cvar_limit=0.08,
        daily_loss_cap=0.05,
    )
    assert not ok
    assert any("max_drawdown" in r for r in reasons)
    assert any("cvar" in r for r in reasons)
    assert any("worst_daily_loss" in r for r in reasons)


def test_apply_guardrails_all_clear():
    ok, reasons = apply_guardrails(
        max_drawdown=0.10,
        cvar=0.03,
        worst_daily_loss=0.02,
        dd_limit=0.20,
        cvar_limit=0.08,
        daily_loss_cap=0.05,
    )
    assert ok
    assert reasons == []

