
for i = 4:6
   for j = 0:5:360
      H = fspecial('motion', 5, j);
      str = sprintf('blur_kernels/kernel_l%d_a%d.png', i, j);
      imwrite(H, str);
   end
end