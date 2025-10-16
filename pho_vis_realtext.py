import os
import glob 
import cv2 


tmp1 = sorted(glob.glob('result_train/fp16_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_msgDiTfeat4/realtext/restored_val/*.png'))
tmp2 = sorted(glob.glob('result_train/fp16_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_msgDiTfeat24/realtext/restored_val/*.png'))

tmp3 = sorted(glob.glob('result_train/fp16_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_msgDiTfeat4/realtext/ocr_result_val/*.jpg'))
tmp4 = sorted(glob.glob('result_train/fp16_dit4sr-testr_1e-05-1e-04_lrbranch-attns_ocrloss0.01_msgDiTfeat24/realtext/ocr_result_val/*.jpg'))

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


    os.makedirs('./vis/realtext', exist_ok=True)
    cv2.imwrite(f'./vis/realtext/{t1_id}.jpg', vis)

print('FINISH!')



# hq = sorted(glob.glob('/media/dataset2/text_restoration/tair_published/real_text/HQ/*.jpg'))
# lq = sorted(glob.glob('/media/dataset2/text_restoration/tair_published/real_text/LQ/*.jpg'))

# dit4sr_q = sorted(glob.glob('results/realtext/dit4sr_q_wllava/sample00/*.png'))
# dit4sr_f = sorted(glob.glob('results/realtext/dit4sr_f_wllava/sample00/*.png'))
# tair = sorted(glob.glob('results/realtext/tair/*.png'))


# imgs = zip(hq, lq, dit4sr_q, dit4sr_f, tair)

# for i, (h, l, dq, df, t) in enumerate(imgs):
#     h_img = cv2.imread(h)
#     l_img = cv2.imread(l)
#     l_img = cv2.resize(l_img, (h_img.shape[1], h_img.shape[0]))
#     dq_img = cv2.imread(dq)
#     df_img = cv2.imread(df)
#     t_img = cv2.imread(t)

#     id = h.split('/')[-1].split('.')[0]
#     vis = cv2.hconcat([l_img, t_img, df_img, dq_img, h_img])
#     cv2.imwrite(f'results_vis/realtext/{id}.png', vis)
#     print(f'{i} saved results_vis/realtext/{id}.png')

# print(f'All done!')