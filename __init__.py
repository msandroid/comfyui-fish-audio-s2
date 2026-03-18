from .fish_audio_s2_node import FishAudioS2DownloadModel, FishAudioS2TTSNode

NODE_CLASS_MAPPINGS = {
    "FishAudioS2TTS": FishAudioS2TTSNode,
    "FishAudioS2DownloadModel": FishAudioS2DownloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FishAudioS2TTS": "Fish Audio S2 Pro TTS",
    "FishAudioS2DownloadModel": "Download Fish Audio S2 Pro Model",
}
