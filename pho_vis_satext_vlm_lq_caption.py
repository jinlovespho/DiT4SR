import os 
import glob 



ids = sorted(glob.glob(f'result_vlm/lq_caption/satext_lv1/satext_lv1_Englishques0/llava_7b/*.txt'))
for id in ids:
    id = id.split('/')[-1].split('.')[0]
    
    
    dataset = 'satext_lv3'
    caption_path = f'result_vlm/lq_caption/{dataset}'
    questions_path = sorted(glob.glob(f'{caption_path}/*{dataset}*'))
    
    
    os.makedirs(f'{caption_path}/vis_vlm_prompt', exist_ok=True)
    with open(f'{caption_path}/vis_vlm_prompt/{id}.txt', 'w') as f:
        f.write(f'[dataset]: {dataset}\n')
        f.write(f'[img_id]: {id}\n\n')
        f.write(f'------------------------------------------------------------------------------------------------------\n')


    for q_path in questions_path:
        
        llava_7b_caps  = sorted(glob.glob(f'{q_path}/llava_7b/*{id}*.txt'))
        llava_13b_caps = sorted(glob.glob(f'{q_path}/llava_13b/*{id}*.txt'))
        qwenvl_3b_caps = sorted(glob.glob(f'{q_path}/qwenvl_3b/*{id}*.txt'))
        qwenvl_7B_caps = sorted(glob.glob(f'{q_path}/qwenvl_7b/*{id}*.txt'))
        assert len(llava_7b_caps) == len(llava_13b_caps) == len(qwenvl_3b_caps) == len(qwenvl_7B_caps)
        # print( len(llava_7b_caps), len(llava_13b_caps), len(qwenvl_3b_caps), len(qwenvl_7B_caps))
        
        llava_cap1  = llava_7b_caps[0]
        llava_cap2 = llava_13b_caps[0]
        qwen_cap1 = qwenvl_3b_caps[0]
        qwen_cap2 = qwenvl_7B_caps[0]
        
        llava_cap1_id  = llava_cap1.split('/')[-1].split('.')[0]
        llava_cap2_id = llava_cap2.split('/')[-1].split('.')[0]
        qwen_cap1_id = qwen_cap1.split('/')[-1].split('.')[0]
        qwen_cap2_id = qwen_cap2.split('/')[-1].split('.')[0]
        assert llava_cap1_id==llava_cap2_id==qwen_cap1_id==qwen_cap2_id
        
        
        llava_cap1  = open(llava_cap1).readline()
        llava_cap2 = open(llava_cap2).readline()
        qwen_cap1 = open(qwen_cap1).readline()
        qwen_cap2 = open(qwen_cap2).readline()
        
        with open(f'{caption_path}/vis_vlm_prompt/{id}.txt', 'a') as f:
            f.write(f'[question path]: {q_path}\n\n')
            f.write(f'[llava_7b]: {llava_cap1}\n\n')
            f.write(f'[llava_13b]: {llava_cap2}\n\n')
            f.write(f'[qwenvl_3b]: {qwen_cap1}\n\n')
            f.write(f'[qwenvl_7b]: {qwen_cap2}\n\n\n')
            f.write(f'------------------------------------------------------------------------------------------------------\n')
            
breakpoint()

print('FINISH !')