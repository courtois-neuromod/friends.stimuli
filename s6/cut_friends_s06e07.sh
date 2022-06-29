
singularity run     -B $PWD:/input     -B $PWD:/output     /data/neuromod/containers/melt.simg     -silent colour:black out=59     /input/s6/friends_s06e07.mkv in=0 out=19683 -mix 60 -mixer luma     colour:black out=240 -mix 59 -mixer luma     -attach-track ladspa.1403 0=-25 1=0.25 2=0.4 3=0.6     -attach-track ladspa.1913 0=17 1=-3 2=0.5     -attach-track volume:-70db end=0db in=0 out=60     -attach-track volume:0db end=-70db in=19623 out=19683     -consumer avformat:/output/s6/friends_s06e07a.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=1500k
        
singularity run     -B $PWD:/input     -B $PWD:/output     /data/neuromod/containers/melt.simg     -silent colour:black out=59     /input/s6/friends_s06e07.mkv in=19503 -mix 60 -mixer luma     colour:black out=240 -mix 59 -mixer luma     -attach-track ladspa.1403 0=-25 1=0.25 2=0.4 3=0.6     -attach-track ladspa.1913 0=17 1=-3 2=0.5     -attach-track volume:-70db end=0db in=0 out=60     -attach-track volume:0db end=-70db in=19803 out=19863     -consumer avformat:/output/s6/friends_s06e07b.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=1500k
        