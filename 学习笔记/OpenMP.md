# OpenMP

#### OpenMP编译指导

1. > parallel指令

   ```c
   #pragma omp parallel
   ```

2. > for指令

   ```c 
   #pragma omp parallel for
   ```

3. > sections和section指令

   ``` c  
   #pragma omp sections [clause[[,] clause]…] { [#pragma omp section] structured block [#pragma omp section] structured block …… }
   ```

4. > single指令

5. > private数据属性

   1. 

   - 将一个或多个变量声明为线程的私有变量。 

   - 每个线程都有它自己的变量私有副本，其他线程无法访问。 

   -  即使在并行区域外有同名的共享变量，共享变量在并行区域内不起任何作用，并 且并行区域内不会操作到外面的共享变量。

   - 并行区域内的private变量和并行区域外同名的变量没有存储关联。

   - 如果需要继承原有共享变量的值，则应使用firstprivate子句。 

   -  如果需要在退出并行区域时将私有变量最后的值赋值给对应的共享变量，则可使 用lastprivate子句。
     ``` c
#pragma omp parallel for firstprivate(k),lastprivate(k)
    
     ```

   

 6. > reduction子句

      - 为变量指定一个操作符，每个线程都会创建reduction变量的私有拷贝，在 OpenMP区域结束处，将使用各个线程的私有拷贝的值通过制定的操作符进行 迭代运算，并赋值给原来的变量。 
      -  语法：recutioin(operator:list) 
      -  operator：+ * - & ^ | && || max min



#### OpenMP运行库历程与环境变量

1. > 设置线程数量，优先级从高到低

   ``` c
   #pragma omp parallel num_thread(8)
   
   omp_set_num_thread(8);
   
   export OMP_NUM_THREADS = 3
   ```

2. > 获取正在使用的线程数量和编号

   ```c
   int omp_get_num_threads(void)：返回当前并行区域中的线程数量。
   
   int omp_get_thread_num(void)：返回值当前并行区域中，当前线程在线程组中的编号。这个编号从0开始。
   ```


3. > 获取程序可以使用的CPU核心数

   ```c
   int omp_get_num_procs(void)：返回值为当前程序可以使用的CPU核数。
   ```

4. > 获取墙上时间

   ```c
   double omp_get_wtime(void)：返回值是以秒为单位的墙上时间。在并行区域开始前和结束后分别调用该函数，并求取两次返回值的差，便可计算出并行执行的时间。
   ```

   