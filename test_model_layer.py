#!/usr/bin/env python3
"""Tests for the model layer: capabilities, payload construction and logging.

The point of these tests is the thing that used to be broken: sending a
parameter the selected model rejects. Anthropic removed the sampling
parameters from Opus 4.7 onwards, so a payload carrying `temperature` (or
`temperature` together with `top_p`, which errors on every Claude 4+ model)
comes back as a 400. Nothing here touches the network - the relevant
definitions are lifted out of geddesghost.py and run against stubs, so the
suite is safe to run without an API key.

    python test_model_layer.py
"""

import ast
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import types

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(level=logging.CRITICAL)

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(("PASS  " if ok else "FAIL  ") + label + ("" if ok else f"\n        got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(label)


def extract(names):
    """Pull the named top-level definitions out of geddesghost.py.

    Importing the module outright would run the whole Streamlit app, so the
    definitions under test are compiled in isolation against stubs instead.
    """
    source = open(os.path.join(SCRIPT_DIR, "geddesghost.py"), encoding="utf-8").read()
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    keep.append(node)
                    break
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    return module


class FakeSessionState(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def load_model_layer():
    module = extract({
        'EFFORT_LEVELS_FULL', 'ANTHROPIC_MODELS', 'UNKNOWN_MODEL_CAPABILITIES',
        'OPENROUTER_EFFORT_LEVELS', 'OPENROUTER_EFFORT_OVERRIDES',
        'DISCOVERY_STATE_KEYS', 'clamp_effort',
        'MODEL_CONFIG', 'MODE_DEPTH', 'EFFORT_DEPTH', 'DEPTH_GUIDANCE',
        'get_model_capabilities', 'resolve_depth', 'format_depth_control',
        'format_usage', 'build_http_session', 'ModelResponse', 'ModelAPIHandler',
    })
    session_state = FakeSessionState()
    # OpenRouter capabilities come from its own listing, so the tests seed the
    # discovery cache the way a live fetch would fill it.
    session_state['discovered_openrouter_models'] = {
        'anthropic/claude-sonnet-5': {
            'display_name': 'Anthropic: Claude Sonnet 5',
            'sampling': False, 'effort_levels': ['low', 'medium', 'high'],
            'supported_parameters': ['max_tokens', 'reasoning'],
            'prompt_price': '0.000002',
        },
        'meta-llama/llama-3.3-70b-instruct': {
            'display_name': 'Meta: Llama 3.3 70B Instruct',
            'sampling': True, 'effort_levels': [],
            'supported_parameters': ['max_tokens', 'temperature', 'top_p'],
            'prompt_price': '0.00000012',
        },
        'openai/gpt-5': {
            'display_name': 'OpenAI: GPT-5',
            'sampling': True, 'effort_levels': ['low', 'medium', 'high'],
            'supported_parameters': ['max_tokens', 'temperature', 'reasoning'],
            'prompt_price': '0.00000125',
        },
    }
    namespace = {
        'st': types.SimpleNamespace(session_state=session_state),
        'os': os, 'json': json, 'requests': requests,
        'HTTPAdapter': HTTPAdapter, 'Retry': Retry,
        'RequestException': requests.exceptions.RequestException,
        'logger': logging.getLogger('test'),
    }
    exec(compile(module, 'geddesghost-extract', 'exec'), namespace)
    return namespace


def config_for(ns, model, provider=None):
    config = json.loads(json.dumps(ns['MODEL_CONFIG']))
    config['timeout'] = tuple(config['timeout'])
    if provider is None:
        provider = 'ollama' if ':' in model else 'anthropic'
    config['current_provider'] = provider
    config['providers'][provider]['model'] = model
    return config


def test_capabilities(ns):
    print("\nCapabilities")
    caps = ns['get_model_capabilities']
    check("current models reject sampling", caps('anthropic', 'claude-opus-5')['sampling'], False)
    check("current models accept effort", 'xhigh' in caps('anthropic', 'claude-sonnet-5')['effort_levels'], True)
    check("older models accept sampling", caps('anthropic', 'claude-sonnet-4-20250514')['sampling'], True)
    check("unrecognised model sends nothing", caps('anthropic', 'claude-future-9'),
          {'display_name': 'claude-future-9', 'sampling': False, 'effort_levels': []})
    check("ollama accepts sampling", caps('ollama', 'cogito:latest')['sampling'], True)


def test_payloads(ns):
    print("\nRequest payloads")

    def payload(model, provider=None, **kwargs):
        handler = ns['ModelAPIHandler'](config_for(ns, model, provider))
        return handler.build_payload("question", system_prompt="persona", **kwargs)

    p = payload('claude-sonnet-5', temperature=0.9, effort='xhigh')
    check("effort model omits temperature", 'temperature' in p, False)
    check("effort model omits top_p", 'top_p' in p, False)
    check("effort model sends output_config", p.get('output_config'), {'effort': 'xhigh'})
    check("system prompt preserved", p.get('system'), 'persona')

    p = payload('claude-sonnet-4-20250514', temperature=0.8, effort='xhigh')
    check("sampling model sends temperature", p.get('temperature'), 0.8)
    check("sampling model never sends top_p alongside it", 'top_p' in p, False)
    check("sampling model omits effort", 'output_config' in p, False)

    p = payload('claude-future-9', temperature=0.8, effort='xhigh')
    check("unrecognised model sends no controls", ('temperature' in p, 'output_config' in p), (False, False))

    p = payload('claude-opus-5', effort='nonsense')
    check("unsupported effort level dropped", 'output_config' in p, False)

    p = payload('cogito:latest', temperature=0.8)
    check("ollama keeps temperature", p['options']['temperature'], 0.8)
    check("ollama keeps top_p", p['options']['top_p'], 0.9)


def test_openrouter(ns):
    print("\nOpenRouter")
    caps = ns['get_model_capabilities']

    check("reasoning-only model rejects sampling",
          caps('openrouter', 'anthropic/claude-sonnet-5')['sampling'], False)
    check("reasoning-only model offers effort",
          caps('openrouter', 'anthropic/claude-sonnet-5')['effort_levels'], ['low', 'medium', 'high'])
    check("sampling-only model offers no effort",
          caps('openrouter', 'meta-llama/llama-3.3-70b-instruct')['effort_levels'], [])
    check("unlisted model sends nothing",
          caps('openrouter', 'some/unlisted-model'),
          {'display_name': 'some/unlisted-model', 'sampling': False, 'effort_levels': []})

    def payload(model, **kwargs):
        handler = ns['ModelAPIHandler'](config_for(ns, model, 'openrouter'))
        return handler.build_payload("question", system_prompt="persona", **kwargs)

    p = payload('anthropic/claude-sonnet-5', effort='high')
    check("openrouter uses chat-completions message shape",
          [m['role'] for m in p['messages']], ['system', 'user'])
    check("system prompt becomes a system message", p['messages'][0]['content'], 'persona')
    check("reasoning effort sent", p.get('reasoning'), {'effort': 'high'})
    check("no temperature on a reasoning-only model", 'temperature' in p, False)

    p = payload('meta-llama/llama-3.3-70b-instruct', temperature=0.85)
    check("sampling model gets temperature", p.get('temperature'), 0.85)
    check("sampling model gets no reasoning", 'reasoning' in p, False)

    # A model offering both: effort drives depth, temperature is not smuggled
    # in from config behind it.
    p = payload('openai/gpt-5', effort='high')
    check("both-capable model sends reasoning", p.get('reasoning'), {'effort': 'high'})
    check("both-capable model omits default temperature", 'temperature' in p, False)
    p = payload('openai/gpt-5', temperature=0.2, effort='high')
    check("explicit temperature still honoured", p.get('temperature'), 0.2)

    p = payload('some/unlisted-model', temperature=0.9, effort='high')
    check("unlisted model sends no controls", ('temperature' in p, 'reasoning' in p), (False, False))

    # Response parsing (OpenAI-shaped)
    handler = ns['ModelAPIHandler'](config_for(ns, 'anthropic/claude-sonnet-5', 'openrouter'))
    result = handler._parse_openrouter_response({
        'model': 'anthropic/claude-sonnet-5',
        'choices': [{'message': {'role': 'assistant', 'content': ' By leaves we live. '},
                     'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 800, 'completion_tokens': 210},
    })
    check("openrouter text extracted", result.text, 'By leaves we live.')
    check("openrouter usage normalised",
          (result.usage['input_tokens'], result.usage['output_tokens']), (800, 210))

    truncated = handler._parse_openrouter_response({
        'choices': [{'message': {'content': 'cut off'}, 'finish_reason': 'length'}],
        'usage': {},
    })
    check("length maps to max_tokens", truncated.stop_reason, 'max_tokens')

    try:
        handler._parse_openrouter_response({'error': {'message': 'upstream is down'}})
        check("error in a 200 body raises", False, True)
    except ValueError as exc:
        check("error in a 200 body raises", 'upstream is down' in str(exc), True)

    try:
        handler._parse_openrouter_response({
            'choices': [{'message': {'content': ''}, 'finish_reason': 'content_filter'}]})
        check("content filter raises", False, True)
    except ValueError as exc:
        check("content filter raises", 'declined' in str(exc), True)


def test_effort_clamping(ns):
    print("\nEffort clamping")
    clamp = ns['clamp_effort']
    full = ns['EFFORT_LEVELS_FULL']
    short = ['low', 'medium', 'high']
    check("exact level kept", clamp('high', full), 'high')
    check("xhigh clamps down to high", clamp('xhigh', short), 'high')
    check("max clamps down to high", clamp('max', short), 'high')
    check("modes keep their ordering", [clamp(e, short) for e in ('medium', 'high', 'xhigh')],
          ['medium', 'high', 'high'])
    check("low survives a top-heavy ladder", clamp('low', ['high', 'max']), 'high')
    check("no ladder means no effort", clamp('high', []), None)
    check("unrecognised value dropped", clamp('nonsense', full), None)


def test_depth(ns):
    print("\nDepth resolution")
    resolve = ns['resolve_depth']
    check("high temperature is expansive", resolve('survey', temperature=0.9), 'expansive')
    check("mid temperature is balanced", resolve('survey', temperature=0.7), 'balanced')
    check("low temperature is focused", resolve('proposition', temperature=0.4), 'focused')
    check("xhigh effort is expansive", resolve('survey', effort='xhigh'), 'expansive')
    check("medium effort is focused", resolve('synthesis', effort='medium'), 'focused')
    check("no control falls back to the mode", resolve('proposition'), 'expansive')


def test_response_parsing(ns):
    print("\nResponse parsing")
    handler = ns['ModelAPIHandler'](config_for(ns, 'claude-sonnet-5'))

    result = handler._parse_anthropic_response({
        'content': [
            {'type': 'thinking', 'thinking': ''},
            {'type': 'text', 'text': 'By leaves we live.'},
        ],
        'stop_reason': 'end_turn',
        'model': 'claude-sonnet-5',
        'usage': {'input_tokens': 120, 'output_tokens': 40, 'cache_read_input_tokens': 0},
    })
    check("text blocks extracted, thinking skipped", result.text, 'By leaves we live.')
    check("usage captured", (result.usage['input_tokens'], result.usage['output_tokens']), (120, 40))

    try:
        handler._parse_anthropic_response({
            'stop_reason': 'refusal',
            'stop_details': {'category': 'cyber'},
            'content': [],
        })
        check("refusal raises rather than returning empty text", False, True)
    except ValueError as exc:
        check("refusal raises rather than returning empty text", 'cyber' in str(exc), True)

    ollama = ns['ModelAPIHandler'](config_for(ns, 'cogito:latest'))
    result = ollama._parse_ollama_response({
        'response': '  local reply  ', 'prompt_eval_count': 10,
        'eval_count': 5, 'done_reason': 'stop',
    })
    check("ollama text normalised", result.text, 'local reply')
    check("ollama usage normalised", (result.usage['input_tokens'], result.usage['output_tokens']), (10, 5))


def test_transport(ns):
    print("\nTransport")
    session = ns['build_http_session']()
    retry = session.get_adapter('https://api.anthropic.com').max_retries
    check("POST is retried", 'POST' in retry.allowed_methods, True)
    check("rate limits are retried", 429 in retry.status_forcelist, True)
    check("overloaded is retried", 529 in retry.status_forcelist, True)
    check("a timeout is configured", ns['MODEL_CONFIG']['timeout'], (10, 120))


def test_formatting(ns):
    print("\nDisplay formatting")
    check("usage rendered", ns['format_usage']({'input_tokens': 10, 'output_tokens': 5, 'cache_read_input_tokens': 0}), '10 in / 5 out')
    check("missing usage rendered", ns['format_usage']({}), 'not reported')
    check("effort label", ns['format_depth_control']({'depth': 'expansive', 'effort': 'xhigh', 'temperature': None}), 'expansive (effort xhigh)')
    check("temperature label", ns['format_depth_control']({'depth': 'balanced', 'temperature': 0.7, 'effort': None}), 'balanced (temperature 0.7)')
    check("prompt-only label", ns['format_depth_control']({'depth': 'focused'}), 'focused (prompt only)')


def test_logging():
    print("\nLogging")
    from admin_dashboard import ResponseEvaluator

    module = extract({'initialize_log_files', 'update_chat_logs', 'MODEL_CONFIG',
                      'MODE_DEPTH', 'EFFORT_DEPTH', 'resolve_depth'})
    tmp = tempfile.mkdtemp()
    try:
        namespace = {
            'st': types.SimpleNamespace(session_state=types.SimpleNamespace(
                response_evaluator=ResponseEvaluator(), cognitive_modes=None)),
            'os': os, 'csv': csv, 'json': json,
            'datetime': __import__('datetime').datetime,
            'script_dir': tmp, 'logger': logging.getLogger('test'),
        }
        exec(compile(module, 'geddesghost-extract', 'exec'), namespace)

        csv_file, json_file = namespace['initialize_log_files']()
        generation_info = {
            'temperature': None, 'effort': 'xhigh', 'depth': 'expansive',
            'source': 'auto (proposition)', 'mode': 'proposition',
            'provider': 'anthropic', 'model': 'claude-sonnet-5',
            'usage': {'input_tokens': 2400, 'output_tokens': 900, 'cache_read_input_tokens': 0},
            'stop_reason': 'end_turn',
        }
        namespace['update_chat_logs'](
            "Rob", "How does a city grow?", "By leaves we live.",
            ["geddes.pdf"], ["geddes.pdf (score: 0.81)"],
            csv_file, json_file, generation_info=generation_info,
        )

        rows = list(csv.DictReader(open(csv_file, encoding='utf-8')))
        check("one row written", len(rows), 1)
        check("effort logged", rows[0]['effort'], 'xhigh')
        check("depth logged", rows[0]['depth'], 'expansive')
        check("input tokens logged", rows[0]['input_tokens'], '2400')
        check("output tokens logged", rows[0]['output_tokens'], '900')
        check("stop reason logged", rows[0]['stop_reason'], 'end_turn')
        check("temperature blank on an effort model", rows[0]['actual_temperature'], '')
        check("header written once", open(csv_file, encoding='utf-8').readline().count('effort'), 1)

        entry = json.load(open(json_file, encoding='utf-8'))[0]
        check("json carries usage", entry['usage']['output_tokens'], 900)
        check("json carries effort", entry['effort'], 'xhigh')

        namespace['update_chat_logs'](
            "Rob", "And again?", "Again.", [], [], csv_file, json_file,
            generation_info=dict(generation_info, effort='medium', depth='focused'),
        )
        rows = list(csv.DictReader(open(csv_file, encoding='utf-8')))
        check("second row appended", len(rows), 2)
        check("second row keeps its columns", rows[1]['effort'], 'medium')
    finally:
        shutil.rmtree(tmp)


def test_evaluator():
    print("\nResponse evaluator")
    from admin_dashboard import ResponseEvaluator

    evaluator = ResponseEvaluator()
    evaluator.evaluate_response("A response about nature and ecology.", "survey", temperature=0.7)
    evaluator.evaluate_response("Another, which could be speculative.", "proposition", effort="xhigh")
    result = evaluator.evaluate_response("No controls at all.", "synthesis")
    keys = set(result['temperature_effectiveness'])
    check("temperature keeps its numeric key", 0.7 in keys, True)
    check("effort gets its own key", 'effort=xhigh' in keys, True)
    check("prompt-only gets its own key", 'prompt-only' in keys, True)


def main():
    ns = load_model_layer()
    test_capabilities(ns)
    test_payloads(ns)
    test_openrouter(ns)
    test_effort_clamping(ns)
    test_depth(ns)
    test_response_parsing(ns)
    test_transport(ns)
    test_formatting(ns)
    test_logging()
    test_evaluator()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
