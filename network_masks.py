import jax


def create_decoder_mask(params, decoder_name="decoder"):
    """Creates boolean mask were any leaves under decoder are set to False."""

    def _mask_fn(path, _):
        def f(key):
            try:
                return key.key
            except:
                return key.name

        # Check if any part of the path contains 'decoder'
        return (
            "frozen" if decoder_name in [str(f(part)) for part in path] else "learned"
        )

    # Create mask using tree_map_with_path
    return jax.tree_util.tree_map_with_path(lambda path, _: _mask_fn(path, _), params)


def create_bias_mask(params):
    """Creates boolean mask were any leaves under decoder are set to False."""

    def _mask_fn(path, _):
        def f(key):
            try:
                return key.key
            except:
                return key.name

        # Check if any part of the path contains 'decoder'
        return "frozen" if "bias" in [str(f(part)) for part in path] else "learned"

    # Create mask using tree_map_with_path
    return jax.tree_util.tree_map_with_path(lambda path, _: _mask_fn(path, _), params)
