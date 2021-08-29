using TestImages, FileIO, ImageCore



function main()
    img = Float32.(channelview(testimage("mandril_color")))

    img_shear = clamp01.(shear(img, 150, 2, 3))
    img_rotated = clamp01.(rotate(img, 50, (2, 3)))
    img_rotated2 = clamp01.(rotate(img_rotated, -50, (2, 3)))


    save("../../paper/figures/img_sheared.png", colorview(RGB, img_shear))
    save("../../paper/figures/img_rotated.png", colorview(RGB, img_rotated))
    save("../../paper/figures/img_rotated2.png", colorview(RGB, img_rotated2))
end


