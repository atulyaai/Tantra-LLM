def test_agent_runs():
    from Training.serve_api import build_agent
    agent = build_agent()
    traces, text = agent.run("What is 2+2? Use calc tool if needed.")
    assert isinstance(text, str)


