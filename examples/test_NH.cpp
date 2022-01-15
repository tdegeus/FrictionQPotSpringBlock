#include <FrictionQPotSpringBlock/Line1d.h>
#include <fmt/core.h>
#include <fstream>
#include <prrng.h>
#include <xtensor/xcsv.hpp>
#include <windows.h>
#pragma comment(lib, "winmm.lib ")

int main()
{
    DWORD r_start = timeGetTime();

    size_t N = 900;

    size_t nchunk = 40000;// size of chunk of yield positions kept in memory
    //size_t nbuffer = 10000;// buffer to keep left

    double input_speed = 0;
    double input_temperature = 0;
    int input_stepsnumber = 30000;
    int input_interval = 100;

    printf("frame speed:");
    scanf("%lf",&input_speed);
    printf("set temperature:");
    scanf("%lf",&input_temperature);
    printf("step numbers:");
    scanf("%d",&input_stepsnumber);
    if (input_stepsnumber > 1000000 || input_stepsnumber < 100000)
    {
        printf("interval:");
        scanf("%d",&input_interval);
    }
    else
    {
        input_interval = 100;
    }
        
    
    

    xt::xtensor<size_t, 1> initstate = xt::arange<size_t>(N);//seed for each particle.
    xt::xtensor<size_t, 1> initseq = xt::zeros<size_t>({N});//somewhat configure prrng random generator.
    auto generators = prrng::auto_pcg32(initstate, initseq);//generate, initializing the seeds.

    xt::xtensor<double, 2> y = 2.0 * generators.random({nchunk});//generate 20000 random number for each particle.
    y = xt::cumsum(y, 1);//convert sequence of random numbers to sequence of positions in potential space.
    y -= 50.0;//shift potential position initially.

    FrictionQPotSpringBlock::Line1d::System sys(N, y);

    sys.set_dt(0.1);
    sys.set_eta(2.0 * std::sqrt(3.0) / 10.0);
    sys.set_m(1);
    sys.set_mu(1.0);
    sys.set_k_neighbours(1.0);
    sys.set_k_frame(1/ double(N));
    sys.set_Q(0.1);
    sys.set_temperature(input_temperature);
    sys.set_kBoltzmann(1);

    sys.initiate_gamma(0.0);
    sys.initiate_lns(0.0);
    sys.initiate_temperature();

    FILE* recorder_inst_temperature;
    recorder_inst_temperature= fopen("inst_temperature.txt","w");
    double record_temperature;

    FILE* recorder_force_frame;
    recorder_force_frame= fopen("force_frame.txt","w");
    double record_force;

    FILE* recorder_force_potential;
    recorder_force_potential= fopen("force_potential.txt","w");
    double record_force_po;

    FILE* recorder_position;
    recorder_position= fopen("position.txt","w");
    double record_position;

    int n_step = input_stepsnumber;

    for (int run_step = 0; run_step < n_step; run_step++)
    {
        sys.NHtimestep(input_speed, nchunk);
        //sys.printStep(run_step);

        if (run_step % input_interval == 0)
        {
            record_temperature = sys.output_temperature();
            record_force = sys.output_force_frame();
            record_force_po = sys.output_force_potential();
            record_position = sys.output_position();
            fprintf(recorder_inst_temperature,"%.3e\n",record_temperature);
            fprintf(recorder_force_frame,"%3f\n",record_force);
            fprintf(recorder_force_potential,"%3f\n",record_force_po);
            fprintf(recorder_position,"%3f\n",record_position);
        }


        /*want to guarantee staying in chunk
        if (xt::any(sys.i_chunk() > nbuffer)) {
            for (size_t p = 0; p < N; ++p) {
                QPot::Chunked& yp = sys.y(p);
                auto nb = yp.size() - yp.i_chunk() + nbuffer;
                if (nb >= nchunk) {
                    continue;
                }
                yp.shift_dy(yp.istop(), xt::eval(2.0 * generators[p].random({nchunk - nb})), nb);
            }
        }
        */
    }

    fclose(recorder_inst_temperature);
    fclose(recorder_force_frame);
    fclose(recorder_force_potential);
    fclose(recorder_position);

    DWORD r_end = timeGetTime();
    printf("running time: %3f",double(r_end - r_start));

    system("pause");

    return 0;
}
