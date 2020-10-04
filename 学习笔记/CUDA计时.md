### CUDA中的计时方法

​	问题描述：一般利用CUDA进行加速处理时,都需要测试CUDA程序的运行时间,来对比得到的加速效果。

​	解决方法:

1. GPU端计时,即设备端计时.
2. GPU端计时,即设备端计时.

设备端计时有两种不同的方法，分别是调用clock()函数和使用CUDA API的事件管理功能



#### clock函数计时

1. 在内核函数中要测量一段代码的开始和结束的位置分别调用一次clock函数，并将结果记录下来。

2. 根据这两次clock函数作为返回值，做差计算，然后除以GPU的运行频率(SP的频率)即可得到内核执行时间。

   一般只需要记录每个block执行需要的时间，最后将得到多个block的开始和结束时间，然后比较这多个开始和结束时间，选择最小的开始(最早开始的block)时间和最大的结束(最晚结束的block)，这两个时间值做差，除以GPU的运行频率即可得到内核执行时间



#### CUDA API时间计时

利用CUDA提供的时间管理api实现计时功能

```c
/使用event计算时间
46     float time_elapsed=0;
47     cudaEvent_t start,stop;
48     cudaEventCreate(&start);    //创建Event
49     cudaEventCreate(&stop);
50 
51     cudaEventRecord( start,0);    //记录当前时间
52     mul<<<blocks, threads, 0, 0>>>(dev_a,NUM);
53     cudaEventRecord( stop,0);    //记录当前时间
54 
55     cudaEventSynchronize(start);    //Waits for an event to complete.
56     cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
57     cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
58 
59     cudaMemcpy(&host_a,dev_a,sizeof(host_a),cudaMemcpyDeviceToHost);    //计算结果回传到CPU
60 
61     cudaEventDestroy(start);    //destory the event
62     cudaEventDestroy(stop);
63     cudaFree(dev_a);//释放GPU内存
64     printf("执行时间：%f(ms)\n",time_elapsed);
65     return 0 ;
```

