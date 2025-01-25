const metadata_text = metadata || "";
const initial_summary = summary || "";

export default `You are analyzing a video summary that needs to be enriched with metadata context.

Given the video's metadata and initial summary, provide a more complete understanding that incorporates the video's origin and purpose.

Guidelines for Recontextualization:
1. Consider the metadata FIRST - it provides crucial context about the video's origin and purpose, but never mention the metadata in your summary (unless it is a title or description of the video).
2. Use the video's title, creator, and other metadata to properly frame the context (without mentioning the metadata), but IGNORE:
   - Software/tool attributions (e.g. "Created with Clipchamp", "Made in Adobe", etc.)
   - Watermarks or branding from editing tools
   - Generic platform metadata (e.g. "Uploaded to YouTube")
   - Encoding related metadata (e.g. "Encoded with x264", "Encoded with x265", etc.)
   - Video resolution (e.g. "1080p", "4K", etc.)
   - Frame rate (e.g. "30fps", "60fps", etc.)
   - Audio codec (e.g. "AAC", "MP3", etc.)
   - Video codec (e.g. "H.264", "H.265", etc.)
   - Video bit rate (e.g. "1000kbps", "4000kbps", etc.)
   - Audio bit rate (e.g. "128kbps", "320kbps", etc.)
3. Pay special attention to:
   - The video's intended purpose (based on meaningful metadata, but without mentioning the metadata)
   - Professional vs. amateur content (you don't have to explicitly mention this)
   - Genre and style implications
4. Maintain objectivity while acknowledging the full context
5. Don't speculate too much beyond what's supported by meaningful metadata
6. Keep roughly the same length and style, 2-4 sentences maximum
7. Your output must include <recontextualized_summary> open tags and </recontextualized_summary> close tags, with the summary content between them. 
8. Don't be too specific or verbose in your summary, keep it general and concise.

Output Format:
<recontextualized_summary>
A well-written, informed, recontextualized summary, that utilizes the title and description metadata if it is relevant. Must start with "This video presents"
</recontextualized_summary>

Input:
<metadata>
${metadata_text}
</metadata>

<initial_summary>
${initial_summary}
</initial_summary>` 