def _make_graphed_buffer_callable(function, example, *, stream=None):
    """Capture a read-only tensor function with one contiguous input buffer.

    The returned callable copies into persistent storage and submits a single
    graph replay. Its result aliases persistent storage until the next replay.
    The caller must consume the result before calling it again and use the same
    host thread and replay stream for consecutive calls.
    """
    if not example.is_npu or not example.is_contiguous() or example.requires_grad:
        raise ValueError("expected one contiguous NPU input without gradients")
    replay_stream = torch_npu.npu.current_stream(example.device)
    replay_thread = threading.get_ident()
    if stream is None:
        stream = torch_npu.npu.Stream(device=example.device)
    static = torch.empty_like(example)
    stream.wait_stream(replay_stream)
    with torch.no_grad(), torch_npu.npu.stream(stream):
        static.copy_(example)
        function(static)
    stream.synchronize()
    captured = NPUGraph()
    with torch.no_grad(), graph(captured, stream=stream):
        result = function(static)
    stream.synchronize()

    def run(value):
        if (value.shape != static.shape or value.dtype != static.dtype or value.device != static.device
                or not value.is_contiguous() or value.requires_grad):
            raise ValueError("graph replay input metadata changed")
        if threading.get_ident() != replay_thread or torch_npu.npu.current_stream(static.device) != replay_stream:
            raise RuntimeError("single-buffer replay must use its original stream")
        static.copy_(value)
        captured.replay()
        return result

    return run
