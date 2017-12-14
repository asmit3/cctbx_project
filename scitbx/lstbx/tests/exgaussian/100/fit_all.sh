for file in `ls exgauss_*`
#export BOOST_ADAPTBX_SIGNALS_DEFAULT=1
#export BOOST_ADAPTBX_FPE_DEFAULT=1 
#for ((i=22; i <= 9817622; i=$i+98000000000000000176))
do
    echo $file
    nohup dials.python ../minimize_exgauss.py $file >> logfile &
done
