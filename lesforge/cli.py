"""lesforge command-line interface.

Subcommands:
    doctor   - self-check the package (backends, forward pass, benchmarks)
    generate - run a single mock generation with LES stats
    train    - train the skill signal and print the hit-rate curve
    bench    - run all benchmarks and print a table
    serve    - start the OpenAI-compatible HTTP server
"""
import argparse

from .config import load_config
from .backends import get_backend
from .train import Trainer
from .bench import run_all, summarize


def cmd_doctor(cfg, args):
    from .doctor import main
    main()


def cmd_generate(cfg, args):
    b = get_backend(cfg.model.name, cfg)
    text, out = b.generate(args.prompt)
    print("prompt :", args.prompt)
    print("answer :", text)
    print("stats  :", out.get("stats"))


def cmd_train(cfg, args):
    b = get_backend(cfg.model.name, cfg)
    tr = Trainer(b, lr=args.lr)
    hist = tr.train(epochs=args.epochs)
    print("%-6s %-8s %-10s" % ("epoch", "skill", "hit_rate"))
    for e, s, h in hist:
        print("%-6d %-8.4f %-10.4f" % (e, s, h))


def cmd_bench(cfg, args):
    results = run_all(cfg, skill=cfg.model.skill)
    print(summarize(results))


def cmd_serve(cfg, args):
    from .serving.server import run_server
    run_server(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lesforge",
                                 description="Lookahead Expert Scheduling for on-device MoE VLMs")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("doctor")
    g = sub.add_parser("generate")
    g.add_argument("prompt", default="describe", nargs="?")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--lr", type=float, default=0.05)
    sub.add_parser("bench")
    sub.add_parser("serve")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd is None or args.cmd == "doctor":
        cmd_doctor(cfg, args)
    elif args.cmd == "generate":
        cmd_generate(cfg, args)
    elif args.cmd == "train":
        cmd_train(cfg, args)
    elif args.cmd == "bench":
        cmd_bench(cfg, args)
    elif args.cmd == "serve":
        cmd_serve(cfg, args)


if __name__ == "__main__":
    main()
