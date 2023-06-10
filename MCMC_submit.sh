# nohup python emri_pe.py -Tobs 4.0 -M 8019425.9054786945 -mu 685.1946995167503 -p0 12.115436197262476 -e0 0.39141517457483055 -dev 7 -eps 1e-5 -dt 10.0 -injectFD 0 -template td -nwalkers 128 -ntemps 1 -downsample 0 > out_td_td.out &
# nohup python emri_pe.py -Tobs 4.0 -M 8019425.9054786945 -mu 685.1946995167503 -p0 12.115436197262476 -e0 0.39141517457483055 -dev 6 -eps 1e-5 -dt 10.0 -injectFD 1 -template fd -nwalkers 128 -ntemps 1 -downsample 0 > out_fd_fd.out &
# nohup python emri_pe.py -Tobs 4.0 -M 8019425.9054786945 -mu 685.1946995167503 -p0 12.115436197262476 -e0 0.39141517457483055 -dev 5 -eps 1e-5 -dt 10.0 -injectFD 0 -template fd -nwalkers 128 -ntemps 1 -downsample 0 > out_fd_td.out &

Tobs=4.0
M=3670041.7362535275
mu=292.0583167470244
p0=13.709101864726545
e0=0.5794130830706371
eps=1e-5
dt=10.0
nwalkers=32
ntemps=1
downsample=0

dev=1
injectFD=0
template="fd"
nohup python emri_pe.py -Tobs $Tobs -M $M -mu $mu -p0 $p0 -e0 $e0 -dev $dev -eps $eps -dt $dt -injectFD $injectFD -template $template -nwalkers $nwalkers -ntemps $ntemps -downsample $downsample -window_flag 1 > out_0.out &

dev=4
injectFD=1
template="fd"
nohup python emri_pe.py -Tobs $Tobs -M $M -mu $mu -p0 $p0 -e0 $e0 -dev $dev -eps $eps -dt $dt -injectFD $injectFD -template $template -nwalkers $nwalkers -ntemps $ntemps -downsample $downsample -window_flag 1 > out_1.out &

dev=5
injectFD=0
template="td"
nohup python emri_pe.py -Tobs $Tobs -M $M -mu $mu -p0 $p0 -e0 $e0 -dev $dev -eps $eps -dt $dt -injectFD $injectFD -template $template -nwalkers $nwalkers -ntemps $ntemps -downsample $downsample -window_flag 1 > out_2.out &

nohup python emri_pe.py -Tobs 4.0 -M 3670041.7362535275 -mu 292.0583167470244 -p0 13.709101864726545 -e0 0.5794130830706371 -eps 1e-2 -dt 10.0 -injectFD 1 -template fd -nwalkers 32 -ntemps 1 -downsample 0 -dev 0 -window_flag 0 > out_4.out &
# dev=5
# injectFD=0
# template="td"
# nohup python emri_pe.py -Tobs $Tobs -M $M -mu $mu -p0 $p0 -e0 $e0 -dev $dev -eps $eps -dt $dt -injectFD $injectFD -template $template -nwalkers $nwalkers -ntemps $ntemps -downsample $downsample > out_2.out &


# dev=5
# injectFD=1
# eps=1e-2
# template="fd"
# nohup python emri_pe.py -Tobs $Tobs -M $M -mu $mu -p0 $p0 -e0 $e0 -dev $dev -eps $eps -dt $dt -injectFD $injectFD -template $template -nwalkers $nwalkers -ntemps $ntemps -downsample $downsample > out_4.out &

# 603401  2954440  2954439