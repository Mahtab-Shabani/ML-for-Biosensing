function H = spectral_entropy(Pxx)
    P = Pxx(:);
    P = P / (sum(P) + eps);
    P(P<=0) = eps;
    H = -sum(P .* log2(P));
    H = H / log2(length(P)); % normalize to [0,1]
end
