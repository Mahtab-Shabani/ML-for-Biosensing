function bp = bandpower_from_psd(Pxx,f,band)
    idx = f >= band(1) & f <= band(2);
    bp = trapz(f(idx), Pxx(idx));
end
