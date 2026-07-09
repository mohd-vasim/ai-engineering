## Dataset summary

- **17 videos**, **100 snapshots** (10s interval), 11 categories
- **food**: 3 videos (Urban Street Food Market, Steak Prep, Pour Over Coffee)
- **nature**: 4 videos (Mountain Sunrise, Ocean Waves, Coastal Cliffs, Flower Blooming)
- **technology**: 2 videos (Robot Arm, Coding/Web App)
- **6 long videos** (60s): video_001, 003, 006, 009, 015, 016
- One single-snapshot occurrence of "bee on a flower" at t=50 in the Flower Blooming timelapse

## Sample queries to try with the agent

### Discovery / metadata
1. "What kinds of videos are in the dataset and how many of each?"
2. "How many videos and total snapshots are stored?"
3. "List all videos in the `nature` category."
4. "Which videos are exactly 60 seconds long?"

### Open-ended content search
5. "Which videos involve cooking or food preparation?"
6. "Show me scenes of a city at night."
7. "Find me moments where a person is working on a hobby project (robotics, coding, music, pottery)."
8. "What videos show an animal?"
9. "Find me any moment with a flying insect on a flower."

### Time-range / video-specific
10. "Walk me through the first 20 seconds of the mountain sunrise video."
11. "What happens at the very end of the coffee brewing video?"
12. "Show me all snapshots of the steak cooking video between 10s and 40s."
13. "Give me a per-second timeline of the robotics assembly video."

### Filter combos
14. "Show me all `food` videos that are at least 50 seconds long."
15. "List clips from the `technology` category whose duration is exactly 10 seconds."
16. "Get all snapshots in the `nature` category that start before 20s."

### Cross-cutting / aggregation
17. "Compare the robot arm assembly to the coding session — what tools/parts does each involve?"
18. "Which videos would be calming to watch and why?"
19. "Group the videos into instructional vs scenic vs action."
20. "If I only have 2 minutes, give me a 5-clip highlight reel spanning different categories."

### Edge cases (good for testing the agent)
21. "Show me videos in the `sci-fi` category." → should return nothing and suggest alternatives.
22. "What happens in the last 5 seconds of video_002?" → tests the end of a non-aligned window.
23. "Tell me about a video I haven't asked about yet." → forces the agent to use `list_video_categories` or sample broadly.