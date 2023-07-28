export filename=$1
export input_video=$2
export input_audio=$3
export frames_wav2lip=$4
export frames_hd=$5
export output_videos_wav2lip=$6
export output_videos_hd=$7
export back_dir=$8
export resize_factor=$9
export face_det_batch_size=${10}

python3 inference.py --checkpoint_path "checkpoints/wav2lip_gan.pth" --segmentation_path "checkpoints/face_segmentation.pth" --sr_path "checkpoints/esrgan_yunying.pth" --face ${input_video}/${filename}.mp4 --audio ${input_audio} --save_frames --gt_path "data/gt" --pred_path "data/lq" --no_sr --no_segmentation --outfile ${output_videos_wav2lip}/${filename}.mp4 --resize_factor ${resize_factor} --face_det_batch_size ${face_det_batch_size}
python video2frames.py --input_video ${output_videos_wav2lip}/${filename}.mp4 --frames_path ${frames_wav2lip}/${filename}
cd Real-ESRGAN
python inference_realesrgan.py -n RealESRGAN_x4plus -i ${back_dir}/${frames_wav2lip}/${filename} --output ${back_dir}/${frames_hd}/${filename} --outscale 3.5 --face_enhance
ffmpeg -r 20 -i ${back_dir}/${frames_hd}/${filename}/frame_%05d_out.jpg -i ${back_dir}/${input_audio} -vcodec libx264 -crf 25 -preset veryslow -acodec copy ${back_dir}/${output_videos_hd}/${filename}.mp4
