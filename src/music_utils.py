import pylast

API_KEY = "5d2566c5242c2095f9767cb89bb16b3b"
API_SECRET = "ede64977050ce1b95aa1a03f6d4bd79a"

network = pylast.LastFMNetwork(
    api_key=API_KEY,
    api_secret=API_SECRET
)

EMOTION_TAGS = {
    "Happy":    "happy",
    "Calm":     "chill",
    "Sad":      "sad",
    "Stressed": "relaxing"
}

def get_tracks_for_emotion(emotion, n=5):
    tag = EMOTION_TAGS[emotion]
    tracks = network.get_tag(tag).get_top_tracks(limit=n)
    result = []
    for track in tracks:
        result.append({
            'name': track.item.title,
            'artist': track.item.artist.name,
            'url': track.item.get_url()
        })
    return result
if __name__ == "__main__":
    print("ThetaPlay - Music Utilities Test")
    print("==================================")
    print("Available emotions: Happy, Calm, Sad, Stressed")
    
    emotion = input("\nEnter emotion: ").strip().capitalize()
    
    if emotion not in EMOTION_TAGS:
        print(f"Invalid emotion. Please choose from: {list(EMOTION_TAGS.keys())}")
    else:
        print(f"\nFetching tracks for '{emotion}'...\n")
        tracks = get_tracks_for_emotion(emotion, n=5)
        for i, track in enumerate(tracks, 1):
            print(f"  {i}. {track['name']} by {track['artist']}")
            print(f"     {track['url']}")