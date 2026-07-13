from npdna import NpDnaAgent, NpDnaCore


def test_agent_defaults_are_local_only():
    agent = NpDnaAgent(NpDnaCore.from_config("seed"))
    assert "cortex_search" in agent.tools
    assert "cortex_store" in agent.tools
    assert "math_eval" in agent.tools
    assert "web_search" not in agent.tools
    assert "code_execute" not in agent.tools
