"""CLI: ``minimal-agent <subcommand> ...``.

Subcommands
-----------

``run``    – one-shot dispatch; optionally sends the result into a Feishu chat
             using the IM Open API (``FEISHU_CHAT_ID`` or ``--chat-id``).
``serve``  – start the long-connection Feishu bot (``lark.ws.Client``);
             receives messages and replies through the IM Open API.

For backwards compatibility, ``minimal-agent "prompt"`` (no subcommand) still
behaves like ``minimal-agent run "prompt"``.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

from .agent import VALID_TOOLS, run
from .log_config import configure_logging, get_logger


log = get_logger(__name__)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("prompt", help="Natural-language prompt for the agent.")
    p.add_argument(
        "-i", "--image",
        help="Optional input image path (jpg/png/webp). Required for it2t and it2i.",
    )
    p.add_argument(
        "-m", "--mode",
        choices=list(VALID_TOOLS),
        help="Force a specific tool. If omitted, the agent routes automatically.",
    )
    p.add_argument(
        "-o", "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "./output_images"),
        help="Directory for generated images (default: %(default)s).",
    )
    p.add_argument(
        "--chat-id",
        default=None,
        help="Feishu chat_id (oc_xxxx) to deliver the result to. "
             "Falls back to $FEISHU_CHAT_ID.",
    )
    p.add_argument(
        "--no-feishu",
        action="store_true",
        help="Do not deliver to Feishu (local run only).",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="minimal-agent",
        description=(
            "Minimal multimodal Feishu agent. Two modes:\n"
            "  run    one-shot dispatch + send to a chat via the IM API\n"
            "  serve  long-connection bot (lark-oapi WebSocket) — receives and replies"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    run_p = sub.add_parser(
        "run",
        help="One-shot dispatch + send to a Feishu chat.",
        description=(
            "Send a single-turn prompt (optionally with an image) to the model "
            "and deliver the result into the chat identified by --chat-id or "
            "$FEISHU_CHAT_ID. Uses the IM API; requires FEISHU_APP_ID and "
            "FEISHU_APP_SECRET. Pass --no-feishu for local-only runs."
        ),
    )
    _add_run_args(run_p)

    sub.add_parser(
        "serve",
        help="Start the long-connection Feishu bot.",
        description=(
            "Connect to Feishu over WebSocket (long connection) and reply to "
            "incoming messages. Requires a self-built app with `im:message`, "
            "`im:message:send_as_bot`, and `im:resource` scopes, and the "
            "im.message.receive_v1 event enabled in long-connection mode. No "
            "public URL / HTTPS / ngrok required."
        ),
    )

    return p


def _dispatch_run(args) -> int:
    log.debug(
        "cli.run.start",
        has_image=bool(args.image),
        mode=args.mode,
        output_dir=args.output_dir,
        no_feishu=args.no_feishu,
        chat_id_override=bool(args.chat_id),
        prompt_len=len(args.prompt or ""),
    )
    try:
        result = run(
            prompt=args.prompt,
            image_path=args.image,
            mode=args.mode,
            output_dir=args.output_dir,
            notify_feishu=not args.no_feishu,
            chat_id=args.chat_id,
        )
    except FileNotFoundError as e:
        log.error("cli.run.input_missing", error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        log.error("cli.run.invalid_args", error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        log.exception("cli.run.unhandled_error", error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not args.no_feishu and not result.feishu_pushed:
        log.warning(
            "cli.run.feishu_delivery_failed",
            tool=result.tool,
            reason=result.feishu_error,
        )
        print(
            "warning: result computed locally but Feishu delivery failed; "
            f"reason: {result.feishu_error}",
            file=sys.stderr,
        )
        return 3
    log.info(
        "cli.run.done",
        tool=result.tool,
        feishu_pushed=result.feishu_pushed,
        has_text=result.text is not None,
        image_path=result.image_path,
    )
    return 0


def _dispatch_serve(args) -> int:  # noqa: ARG001
    # Import lazily so `run` doesn't import the SDK's WebSocket stack.
    from .bot import start as _start
    log.debug("cli.serve.start")
    try:
        _start()
    except SystemExit as e:
        log.error("cli.serve.systemexit", error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        log.info("cli.serve.keyboard_interrupt")
        return 0
    except Exception as e:  # noqa: BLE001
        log.exception("cli.serve.bot_connection_failed", error=str(e))
        print(f"error: bot connection failed: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    load_dotenv()
    # ``configure_logging`` must run *after* ``load_dotenv`` so that
    # ``LOG_LEVEL`` / ``LOG_FORMAT`` / ``FEISHU_DEBUG`` defined in the
    # user's ``.env`` are honored on the very first log line.
    configure_logging()
    argv = list(sys.argv[1:] if argv is None else argv)

    # Back-compat shim: if the first non-flag token is neither "run" nor
    # "serve", insert "run" so old invocations keep working.
    if argv and argv[0] not in ("run", "serve", "-h", "--help"):
        argv = ["run", *argv]

    args = build_parser().parse_args(argv)
    log.debug("cli.dispatch", cmd=args.cmd or "run")

    if args.cmd == "serve":
        return _dispatch_serve(args)
    # Default to run.
    return _dispatch_run(args)


if __name__ == "__main__":
    sys.exit(main())

