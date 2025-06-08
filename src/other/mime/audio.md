# 音乐文件
音乐文件的格式主要分为两种文件，一种是音频数据文件，一种是乐器指令文件。

## 音乐格式
浏览器原生支持主流的声音编码格式

- 普通音乐
  - mp3 (MPEG-1 Audio Layer III)。几乎所有现代浏览器都支持，包括 Chrome、Firefox、Safari、Edge 和 Opera。在高比特率（如 256kbps 或 320kbps）下音质非常好。文件大小：文件大小适中，有损压缩使得文件较小，适合网络传输。使用场景：音乐、播客、音频书籍、网站背景音乐等。
  - aac (Advanced Audio Coding)。主要浏览器都支持，音质：比 mp3 更高效，同样比特率下音质更好。文件大小：与 mp3 相比 aac 文件通常更小且音质更好。使用场景：流媒体服务、在线音乐播放、视频音频。
- 高级音质
  - wav (Waveform Audio File Format)。几乎所有现代浏览器都支持，无压缩格式，音质高，文件体积大，适用于专业音频制作、编辑、存储高质量音频。
  - flac (Free Lossless Audio Codec)。浏览器不支持，无损压缩格式，音质高，文件体积比 wav 小。用途：高保真音频存储和传输。
- 特殊特性
  - opus（Opus Interactive Audio Codec）。现代浏览器支持较好，Safari 支持较差。专门为 Web 设计的音频格式，由 Xiph.Org 基金会开发，之后由互联网工程任务组（IETF）进行标准化。高效的有损压缩格式，适用于语音和音乐。文件大小：非常小，适合实时传输和低延迟应用。使用场景：VoIP、实时通讯、游戏音效、流媒体。opus 格式是一个开放格式，使用上没有任何专利或限制。
  - midi(Musical Instrument Digital Interface)。不包含音频数据，而是包含乐器指令。用于乐器控制、音乐制作、电子乐器。

## 音乐创作软件
- [Steinberg Cubase](https://www.steinberg.net/cubase/)。专业级音乐制作软件。用于电子音乐创作。

## 音乐编辑软件
- [Audacity](https://www.audacityteam.org/)。功能强大的开源音频编辑软件，使用 Qt 进行编写的跨平台软件，支持多轨录音和编辑、效果处理、格式转换等。
- [Pro Tools](https://www.avid.com/pro-tools)。Pro Tools 是行业内使用最广泛的音频工作站，提供了最高水平的音频录音、编辑、混音和后期处理工具。
