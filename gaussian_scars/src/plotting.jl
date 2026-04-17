function mytheme()
    set_theme!(Theme(
        fonts = Attributes(
            regular = "Latin Modern Roman",
            bold = "Latin Modern Roman",
            #bold = "Latin Modern Roman Bold", # Uncomment to get bolded label ticks
            #italic = "Latin Modern Roman Italic",
            #title = "Latin Modern Roman Bold",
            #ticks = "Latin Modern Roman",
        ),
        fontsize = 36,
        Axis = (
            # frame configuration
            spinewidth = 0.8,

            # xticks configuration
            xticksize = 6, #major ticks
            xminorticksize = 3, #minor ticks
            xtickwidth = 0.8, 
            xminortickwidth = 0.6,
            xtickcolor = :black,
            xminortickcolor = :black,
            xtickalign = 1, # 1 = inward, 0, outward, 0.5 = centered
            xminortickalign = 1,
            xticksmirrored = true, # mirror ticks on top
            xminorticksmirrored = true,
            xminorticksvisible = true, # show minor ticks

            # yticks configuration
            yticksize = 6, #major ticks
            yminorticksize = 3, #minor ticks
            ytickwidth = 0.8,
            yminortickwidth = 0.6,
            ytickcolor = :black,
            yminortickcolor = :black,
            ytickalign = 1, # 1 = inward, 0, outward, 0.5 = centered
            yminortickalign = 1,
            yticksmirrored = true, # mirror ticks on right
            yminorticksmirrored = true,
            yminorticksvisible = true, # show minor ticks

            # labels
            xlabelsize = 20,
            ylabelsize = 20,
            xticklabelsize = 20,
            yticklabelsize = 20,

            # grid
            xgridvisible = false,
            ygridvisible = false,
            # xgridcolor = :gray,
            # ygridcolor = :gray,
            # xgridstyle = :dash,
            # ygridstyle = :dash,
        ),
    ))
end