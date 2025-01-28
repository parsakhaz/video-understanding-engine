def get_prompt() -> str:
    return """You are tasked with synthesizing multiple summaries and transcript segments from different parts of a single video into one coherent, comprehensive summary of the entire video.
    The frame and scene summaries, along with transcript segments, are presented in chronological order. This video does not contain photos, any mentions of photos are incorrect and hallucinations.
    
    Output a single overall summary of all the frames, scenes, and dialogue that:
    1. Always starts with "The video presents" to maintain consistent style.
    2. Captures the main narrative arc, inferring the most likely high level overview and theme with the given information, along with emotions and feelings.
    3. Maintains a clear and concise flow
    4. Is roughly the same length as one of the input summaries
    5. Ends immediately after the summary (no extra text)
    6. IMPORTANT: Never refers to frames as "stills" or "images" - they are frames from a continuous sequence
    7. You must focus on observable events and context without making assumptions about who is doing what
    8. You must use neutral language and avoids attributing actions unless explicitly clear
    
    Input Format:
    <chunk_summaries>
    [Chronological summaries of segments]
    </chunk_summaries>

    <transcript>
    [Chronological transcript segments]
    </transcript>""" 