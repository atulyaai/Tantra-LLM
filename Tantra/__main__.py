"""Entry dispatcher.

    python -m Tantra <main-CLI-command>          routing CLI (Tantra.main)
    python -m Tantra <entry_name> [args...]      named CLI entry from a host module

Entry names let the flattened hosts keep their per-task command surfaces:
workbench_cli, auto_growth_cli, sources_cli, prepare_corpus_cli, pilot_cli,
continue_cli, job_cli, candidates_cli, migrate_cli, vjob_cli,
router_eval_cli, reply_eval_cli, review_cli, sheets_cli, instruct_cli,
quality_cli, conversation_sources_cli.
"""
import importlib
import sys

ENTRIES = {
    'workbench_cli': ('train', 'workbench_cli'),
    'bench_peer_cpu': ('bench_cpu', 'bench_peer_cpu'),
    'auto_growth_cli': ('train', 'auto_growth_cli'),
    'sources_cli': ('pretrain', 'sources_cli'),
    'prepare_corpus_cli': ('pretrain', 'prepare_corpus_cli'),
    'pilot_cli': ('pretrain', 'pilot_cli'),
    'continue_cli': ('pretrain', 'continue_cli'),
    'job_cli': ('pretrain', 'job_cli'),
    'candidates_cli': ('tokenizer', 'candidates_cli'),
    'migrate_cli': ('tokenizer', 'migrate_cli'),
    'vjob_cli': ('tokenizer', 'vjob_cli'),
    'router_eval_cli': ('evaluate', 'router_eval_cli'),
    'reply_eval_cli': ('evaluate', 'reply_eval_cli'),
    'review_cli': ('dashboard', 'review_cli'),
    'sheets_cli': ('dashboard', 'sheets_cli'),
    'instruct_cli': ('dashboard', 'instruct_cli'),
    'dashboard': ('dashboard', 'main'),
    'quality_cli': ('conversation_train', 'quality_cli'),
    'conversation_sources_cli': ('conversation_train', 'conversation_sources_cli'),
}


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ENTRIES:
        module_name, function = ENTRIES[argv[0]]
        module = importlib.import_module('Tantra.' + module_name)
        sys.argv = [sys.argv[0], *argv[1:]]
        sys.exit(getattr(module, function)())
    from Tantra.main import main as cli
    sys.exit(cli())


if __name__ == '__main__':
    main()
