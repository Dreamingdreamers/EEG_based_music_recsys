import pylast

# Replace with your actual keys
API_KEY = "5d2566c5242c2095f9767cb89bb16b3b"
API_SECRET = "ede64977050ce1b95aa1a03f6d4bd79a"

network = pylast.LastFMNetwork(
    api_key=API_KEY,
    api_secret=API_SECRET
)

# Test - get tracks for each emotion
EMOTION_TAGS = {
    "Happy":    "happy",
    "Calm":     "chill",
    "Sad":      "sad",
    "Stressed": "relaxing"
}

print("Testing ThetaPlay - Last.fm connection...\n")

for emotion, tag in EMOTION_TAGS.items():
    print(f"── {emotion} ──")
    tracks = network.get_tag(tag).get_top_tracks(limit=3)
    for track in tracks:
        print(f"  {track.item.title} by {track.item.artist.name}")
    print()

print("Last.fm API working successfully!")