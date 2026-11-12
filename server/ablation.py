"""Per-request ablation: one engine process serves stock and ablated requests.

The client selects the mode with the request's model name (`<served id>-abliterated`, or the same form of
serve --model-name when the operator names a second public model) or with the body field "crucible_ablate": true.
The server then adds "ablate": 1 to every engine request object of that turn, so the engine's slot, checkpoint,
frontier and disk matches keep the two modes apart (engine/runtime/tenant.h, the derived state tenant of
the per-request abliteration design). A stock request carries no such field: its engine payload, and therefore every cache and
disk key, stays as it is today. A stock engine refuses the ablated forms with a 400 (never a silent stock answer).
"""
# The suffix of an ablated model name.
ABLATED_SUFFIX = "-abliterated"
# The client's body field (the gateway sends this one), and the engine's request field with its only accepted value.
REQUEST_FIELD = "crucible_ablate"
ENGINE_FIELD = "ablate"
ENGINE_VALUE = 1
# The ready event field of an engine that was started in request mode (CRUCIBLE_ABLATE_MODE=request).
READY_FIELD = "ablate_request"

# The 400 of an ablated request on an engine that serves stock only.
NO_REQUEST_MODE = ("The ablated mode needs an engine that serves ablated requests on demand "
                   "(CRUCIBLE_ABLATE_MODE=request). This engine serves the stock model only.")


def ablated_names(model, model_name=""):
    """The model names that select the ablated mode: the served id's, and the public model name's when it differs."""
    names = [f"{model}{ABLATED_SUFFIX}"]
    if model_name and model_name != model:
        names.append(f"{model_name}{ABLATED_SUFFIX}")
    return tuple(names)


def engine_fields():
    """The engine keys of one ablated request object. Empty for stock, which must keep today's bytes."""
    return {ENGINE_FIELD: ENGINE_VALUE}
