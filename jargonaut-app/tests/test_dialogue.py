"""The conversation layer (help, clarification, follow-ups) with the models replaced by stubs."""
from jargonaut.extractor import Extraction, Span
from jargonaut.pipeline import Answer, Conversation
from jargonaut.router import Route


class StubSystem:
    def __init__(self, routes):
        self.routes, self.asked, self.explained = routes, [], []

    def route(self, question):
        return self.routes.get(question, Route(None, "out_of_scope"))

    def answer_category(self, contract, category, route=None):
        self.asked.append(category)
        best = Span(9.0, "best", 0, 4, "c_0000", 1)
        more = Span(7.0, "second", 10, 16, "c_0001", 2)
        return Answer("answer", "rewrite", category=category), Extraction(best, True, 6.25, 0.9, [more])

    def explain(self, contract, span, extraction, base, timings=None):
        self.explained.append(span.text)
        return Answer("answer", "rewrite", category=base["category"])


def test_help_only_for_whole_messages():
    system = StubSystem({"help me understand the insurance clause": Route("Insurance", "gemma")})
    c = Conversation(system, contract=None)
    assert c.reply("hi").kind == "help" and c.reply("Help?").kind == "help"
    assert c.reply("help me understand the insurance clause").category == "Insurance"


def test_anything_else_pages_through_the_active_topic():
    system = StubSystem({"Which law applies?": Route("Governing Law", "gemma"),
                         "more about insurance": Route("Insurance", "gemma")})
    c = Conversation(system, contract=None)
    assert c.reply("anything else?").kind == "out_of_scope"     # nothing active yet: routed like any question
    c.reply("Which law applies?")
    assert c.reply("Anything else?").kind == "answer" and system.explained == ["second"]
    assert c.reply("more").kind == "no_evidence"                 # no further passages
    assert c.reply("more about insurance").category == "Insurance" and system.asked[-1] == "Insurance"


def test_clarification_then_choice():
    torn = Route(None, "clarify", options=["Non-Compete", "Exclusivity"])
    system = StubSystem({"Can I sell other brands?": torn})
    c = Conversation(system, contract=None)
    a = c.reply("Can I sell other brands?")
    assert a.kind == "clarify" and [o["category"] for o in a.options] == ["Non-Compete", "Exclusivity"]
    assert c.reply("2").category == "Exclusivity"
    c.reply("Can I sell other brands?")
    assert c.reply("exclusive dealing").category == "Exclusivity"   # answering with the topic's name
