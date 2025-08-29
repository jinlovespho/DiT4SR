import glob 
import cv2 


lq = sorted(glob.glob('/media/dataset2/text_restoration/SAMText_test_degradation/lv3/*.jpg'))
hq = sorted(glob.glob('/media/dataset2/text_restoration/100K/test/*.jpg'))

tair_null = sorted(glob.glob('results/satext/lv3/tair_nullprompt/*.png'))
tair = sorted(glob.glob('results/satext/lv3/tair/*.png'))
tair_gt = sorted(glob.glob('results/satext/lv3/tair_gtprompt/*.png'))
dit4sr_q_null = sorted(glob.glob('results/satext/lv3/dit4sr_q_wllava_nullprompt/sample00/*.png'))
dit4sr_q_llava = sorted(glob.glob('results/satext/lv3/dit4sr_q_wllava/sample00/*.png'))
dit4sr_q_gt = sorted(glob.glob('results/satext/lv3/dit4sr_q_wllava_gtprompt/sample00/*.png'))


imgs = zip(lq, hq, tair_null, tair, tair_gt, dit4sr_q_null, dit4sr_q_llava, dit4sr_q_gt)

for i, (l, h, tn, t, tg, dn, dl, dg) in enumerate(imgs):
    h_img = cv2.imread(h)
    l_img = cv2.imread(l)
    l_img = cv2.resize(l_img, (h_img.shape[1], h_img.shape[0]))
    tn_img = cv2.imread(tn)
    t_img = cv2.imread(t)
    tg_img = cv2.imread(tg)
    dn_img = cv2.imread(dn)
    dp_img = cv2.imread(dl)
    dg_img = cv2.imread(dg)
    
    id = h.split('/')[-1].split('.')[0]


    vis1 = cv2.hconcat([l_img, tn_img, t_img, tg_img,])
    vis2 = cv2.hconcat([h_img, dn_img, dp_img, dg_img])
    vis = cv2.vconcat([vis1, vis2])
    cv2.imwrite(f'results_vis/satext_lv3/{id}.png', vis)
    print(f'{i} saved results_vis/satext_lv3/{id}.png')

print(f'All done!')