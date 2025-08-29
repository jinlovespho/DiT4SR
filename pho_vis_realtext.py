import glob 
import cv2 


hq = sorted(glob.glob('/media/dataset2/text_restoration/tair_published/real_text/HQ/*.jpg'))
lq = sorted(glob.glob('/media/dataset2/text_restoration/tair_published/real_text/LQ/*.jpg'))

dit4sr_q = sorted(glob.glob('results/realtext/dit4sr_q_wllava/sample00/*.png'))
dit4sr_f = sorted(glob.glob('results/realtext/dit4sr_f_wllava/sample00/*.png'))
tair = sorted(glob.glob('results/realtext/tair/*.png'))


imgs = zip(hq, lq, dit4sr_q, dit4sr_f, tair)

for i, (h, l, dq, df, t) in enumerate(imgs):
    h_img = cv2.imread(h)
    l_img = cv2.imread(l)
    l_img = cv2.resize(l_img, (h_img.shape[1], h_img.shape[0]))
    dq_img = cv2.imread(dq)
    df_img = cv2.imread(df)
    t_img = cv2.imread(t)

    id = h.split('/')[-1].split('.')[0]
    vis = cv2.hconcat([l_img, t_img, df_img, dq_img, h_img])
    cv2.imwrite(f'results_vis/realtext/{id}.png', vis)
    print(f'{i} saved results_vis/realtext/{id}.png')

print(f'All done!')