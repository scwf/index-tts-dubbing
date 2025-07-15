from indextts.infer import IndexTTS  
  
# 初始化TTS模型  
tts = IndexTTS(model_dir="model-dir/index_tts", cfg_path="model-dir/index_tts/config.yaml")  
  
# 参考音频文件路径  
voice = "refer_voice/qjc_short.WAV"  
  
# 要合成的文本  
text = "这是一段用于语音克隆的示例文本。"  
  
# 输出路径  
output_path = "output/index_output_cloned.wav"  
output_path_fast = "output/index_output_cloned_fast.wav"  

# 执行语音合成,使用普通推理模式  
tts.infer(voice, text, output_path)

  
# 使用快速推理模式，适合长文本处理  
output = tts.infer_fast(  
    audio_prompt=voice,   
    text=text,   
    output_path=output_path_fast 
)