import os
import glob 
import cv2 

dataset = 'satext_lv3'

tmp1 = sorted(glob.glob(f'result_train/fp16_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_msgDiTfeat4/{dataset}/restored_val/*.png'))
tmp2 = sorted(glob.glob(f'result_train/fp16_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_msgDiTfeat24/{dataset}/restored_val/*.png'))

tmp3 = sorted(glob.glob(f'result_train/fp16_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_msgDiTfeat4/{dataset}/ocr_result_val/*.jpg'))
tmp4 = sorted(glob.glob(f'result_train/fp16_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_msgDiTfeat24/{dataset}/ocr_result_val/*.jpg'))

imgs = zip(tmp1, tmp2, tmp3, tmp4)
for t1, t2, t3, t4 in imgs:
    t1_id = t1.split('/')[-1].split('.')[0].split('restored')[-1][1:]
    t2_id = t2.split('/')[-1].split('.')[0].split('restored')[-1][1:]
    t3_id = t3.split('/')[-1].split('.')[0].split('ocr')[-1][1:]
    t4_id = t4.split('/')[-1].split('.')[0].split('ocr')[-1][1:]
    assert t1_id==t2_id==t3_id==t4_id

    t1 = cv2.imread(t1)
    t2 = cv2.imread(t2)
    t3 = cv2.imread(t3)
    t4 = cv2.imread(t4)

    vis1 = cv2.hconcat([t1, t3])
    vis2 = cv2.hconcat([t2, t4])
    vis = cv2.vconcat([vis1, vis2])


    os.makedirs(f'./vis/{dataset}', exist_ok=True)
    cv2.imwrite(f'./vis/{dataset}/{t1_id}.jpg', vis)

print('FINISH!')


# import glob 
# import cv2 


# lq = sorted(glob.glob('/media/dataset2/text_restoration/SAMText_test_degradation/lv3/*.jpg'))
# hq = sorted(glob.glob('/media/dataset2/text_restoration/100K/test/*.jpg'))

# tair_null = sorted(glob.glob('results/satext/lv3/tair_nullprompt/*.png'))
# tair = sorted(glob.glob('results/satext/lv3/tair/*.png'))
# tair_gt = sorted(glob.glob('results/satext/lv3/tair_gtprompt/*.png'))
# dit4sr_q_null = sorted(glob.glob('results/satext/lv3/dit4sr_q_wllava_nullprompt/sample00/*.png'))
# dit4sr_q_llava = sorted(glob.glob('results/satext/lv3/dit4sr_q_wllava/sample00/*.png'))
# dit4sr_q_gt = sorted(glob.glob('results/satext/lv3/dit4sr_q_wllava_gtprompt/sample00/*.png'))


# imgs = zip(lq, hq, tair_null, tair, tair_gt, dit4sr_q_null, dit4sr_q_llava, dit4sr_q_gt)

# for i, (l, h, tn, t, tg, dn, dl, dg) in enumerate(imgs):
#     h_img = cv2.imread(h)
#     l_img = cv2.imread(l)
#     l_img = cv2.resize(l_img, (h_img.shape[1], h_img.shape[0]))
#     tn_img = cv2.imread(tn)
#     t_img = cv2.imread(t)
#     tg_img = cv2.imread(tg)
#     dn_img = cv2.imread(dn)
#     dp_img = cv2.imread(dl)
#     dg_img = cv2.imread(dg)
    
#     id = h.split('/')[-1].split('.')[0]


#     vis1 = cv2.hconcat([l_img, tn_img, t_img, tg_img,])
#     vis2 = cv2.hconcat([h_img, dn_img, dp_img, dg_img])
#     vis = cv2.vconcat([vis1, vis2])
#     cv2.imwrite(f'results_vis/satext_lv3_trained/{id}.png', vis)
#     print(f'{i} saved results_vis/satext_lv3_trained/{id}.png')

# print(f'All done!')


# import glob 
# import cv2 

# # lq
# tmp1 = sorted(glob.glob('/media/dataset2/text_restoration/SAMText_test_degradation/lv3/*.jpg'))
# # hq
# tmp2 = sorted(glob.glob('/media/dataset2/text_restoration/100K/test/*.jpg'))
# # tair llava13b baseline
# tmp3 = sorted(glob.glob('results/satext/lv3/tair/*.png'))
# # dit4sr llav13b baseline
# tmp4 = sorted(glob.glob('results/satext/lv3/dit4sr_q_llavaprompt/sample00/*.png'))
# # dit4sr llava13b ckpt10,000
# tmp5 = sorted(glob.glob('results/satext/lv3/tair_dit4sr_llavaprompt_all_ckpt10000/sample00/*.png'))
# # dit4sr qwen7b baseline
# tmp6 = sorted(glob.glob('results/satext/lv3/dit4sr_q_qwen7prompt/sample00/*.png'))
# # dit4sr qwen7b ckpt10,000
# tmp7 = sorted(glob.glob('results/satext/lv3/dit4sr_q_qwen7prompt_ckpt10000/sample00/*.png'))
# # dit4sr gt
# tmp8 = sorted(glob.glob('results/satext/lv3/dit4sr_q_gtprompt/sample00/*.png'))


# imgs = zip(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8)

# for i, (t1,t2,t3,t4,t5,t6,t7,t8) in enumerate(imgs):
#     h_img = cv2.imread(t2)
#     l_img = cv2.imread(t1)
#     l_img = cv2.resize(l_img, (h_img.shape[1], h_img.shape[0]))
#     t3_img = cv2.imread(t3)
#     t4_img = cv2.imread(t4)
#     t5_img = cv2.imread(t5)
#     t6_img = cv2.imread(t6)
#     t7_img = cv2.imread(t7)
#     t8_img = cv2.imread(t8)
    
#     id = t2.split('/')[-1].split('.')[0]

#     vis1 = cv2.hconcat([l_img, t3_img, t4_img, t5_img])
#     vis2 = cv2.hconcat([t6_img, t7_img, t8_img, h_img])
#     vis = cv2.vconcat([vis1,vis2])
    

#     # vis1 = cv2.hconcat([l_img, tn_img, t_img, tg_img,])
#     # vis2 = cv2.hconcat([h_img, dn_img, dp_img, dg_img])
#     # vis = cv2.vconcat([vis1, vis2])
#     cv2.imwrite(f'results_vis/satext_lv3_trained/{id}.png', vis)
#     print(f'{i} saved results_vis/satext_lv3_trained/{id}.png')

# print(f'All done!')