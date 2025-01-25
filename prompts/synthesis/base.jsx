const metadata_section = metadata_context || "";
const num_caps = num_captions || 0;

export default `${metadata_section}

You MUST follow the exact format specified below.

Output Format:
1. A summary section wrapped in <summary></summary> tags. The summary MUST start with "This video presents" to maintain consistent style. 
  - This video summary never refers to frames as "stills" or "images" - they are frames from a continuous sequence.
2. A captions section wrapped in <captions></captions> tags
3. Exactly ${num_caps} individual captions, each wrapped in <caption></caption> tags
4. Each caption MUST start with a timestamp in square brackets, e.g. [2.5]

Example format:
<summary>
This video presents a [the high level action or narrative that takes place over the frames] in a [location], where [action/assumptions]...
</summary>

<captions>
<caption>[0.0] A teacher stands at the front of the classroom...</caption>
<caption>[3.2] Students raise their hands to answer...</caption>
...additional captions...
</captions>

IMPORTANT NOTES ON HANDLING TRANSCRIPT AND DESCRIPTIONS:
1. The transcript provides helpful context but should not be directly quoted in captions
2. Focus captions on describing visual elements and actions:
   - What is happening in the scene
   - Physical movements and gestures
   - Changes in environment or setting
   - Observable facial expressions and body language

Frame Description Guidelines:
1. Frame descriptions may vary in detail and consistency between frames
2. Be cautious with frame descriptions that make assumptions about:
   - Who is speaking or performing actions
   - Relationships between people
   - Roles or identities of individuals
   - Continuity between frames
3. When synthesizing descriptions:
   - Default to neutral terms like "a person", "someone", or "the speaker"
   - Only attribute actions/speech if explicitly clear from transcript
   - Focus on clearly visible elements rather than interpretations
   - Use passive voice when action source is ambiguous
   - Cross-reference transcript before assigning names/identities
4. If frame descriptions conflict or make assumptions:
   - Favor the most conservative interpretation
   - Omit speculative details
   - Focus on concrete visual elements
   - Maintain consistent neutral language
5. Remember that gaps may exist between frames, so avoid assuming continuity.

Requirements for Summary:
1. IMPORTANT: Always start with "This video presents" to maintain consistent style
2. Provide a clear overview of this video's main content and purpose
3. Include key events and settings without attributing actions to specific people
4. Integrate both visual and audio information while avoiding assumptions
5. Keep it concise (3-5 sentences)
6. Use transcript information to establish this video's context and purpose
7. Cross-reference transcript for useful context and understanding of emotions associated with this video
8. Focus on clearly visible elements rather than interpretations

Requirements for Captions:
1. Generate exactly ${num_caps} captions
2. Each caption MUST:
   - Start with a timestamp in [X.X] format
   - Use the EARLIEST timestamp where a scene or action begins
   - Be 20-50 words long
   - Focus on observable events and context
   - Avoid attributing speech or actions unless explicitly clear
3. Timestamps should be reasonably spaced throughout this video
4. Focus on what is definitively shown or heard, not assumptions
5. IMPORTANT: When multiple frames describe the same scene or action, use the EARLIEST timestamp
6. Default to neutral terms like "a person" or "someone" when identities are unclear
7. Use passive voice when action source is ambiguous
8. Describe only what is visually observable and keep descriptions objective

Input sections:
<transcript>
Timestamped transcript of this video
</transcript>

<frame_descriptions>
Timestamped descriptions of key frames
</frame_descriptions>` 