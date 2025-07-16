
我使用resnet34當作backbone，這個架構屬於U-net
因為我是使用timm直接引用的，所以有興趣也可以試看看其他backbone

訓練過程中有使用隨機的亮度與對比(溫和)，使得最後測試能有更具泛化能力

如果你要訓練，請按照以下步驟
請先至下方連結下載資料集，解壓縮後按照直接把資料夾放在根目錄，應該會長這樣(我幫妳們上傳好了，這玩意原本在百度上)
https://drive.google.com/file/d/1vCNx-OHVqf_rK0JSynlsNSSr2-kjptYU/view?usp=drive_link

 ./data/self-make-Unet/
    image_val
    images
    mask_val
    masks


訓練就直接執行  main.py

測試請執行  test.py 並放你想放的照片
我預設放了我自己的照片 me.jpg


