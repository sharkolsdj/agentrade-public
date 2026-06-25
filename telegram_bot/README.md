# telegram_bot/

This directory holds the Telegram interface used for the manual approval step.
For each proposed trade the operator receives a message with APPROVE, BLOCK,
and MODIFY actions and a fixed approval window; MODIFY allows tightening the
stop, reducing size, or cancelling.

The bot implementation is operational code and is intentionally not included
in this public repository. The approval gate it serves is documented in the
accompanying paper.
