import youtube_dl # MUST HAVE youtube-dl cli installed

def downloadYoutube(url, file_name):
	ydl_opts = {
	    'format': 'bestaudio/best',
	    'postprocessors': [{
	    'key': 'FFmpegExtractAudio',
	    'preferredcodec': 'wav',
	    'preferredquality': '192',
	    }],
	    'outtmpl': file_name,
	}

	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
	    ydl.download([url]) 
