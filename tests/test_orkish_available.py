"""Orkun's hard dependency is an importable Orkish. Fail loudly if it isn't."""


def test_orkish_core_symbols_importable():
    from infer.executors import Sandbox, execute  # noqa: F401
    from infer.monkey_wire import Conversation, ToolCall, parse_calls, serialize  # noqa: F401
    from scripts.verifier import CHECKS, Task, run_trajectory  # noqa: F401
    from scripts.self_play import play_task  # noqa: F401
    from model.tokenizer import OrkishTokenizer  # noqa: F401
    from torch_impl.model.waaagh import WaaaghConfig, WaaaghNet  # noqa: F401


def test_orkun_package_importable():
    import orkun  # noqa: F401
