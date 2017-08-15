package com.cacheserverdeploy.deploy;
import java.util.*;


public class Deploy {

    public static String[] deployServer(String[] graphContent) {

        /** do your work here **/
        return display(graphContent);
    }

    public static Graph read(String[] graphContent) {

        String[] data = graphContent[0].split(" ");
        int server_cost = Integer.parseInt(graphContent[2]);
        Graph G = new Graph(Integer.parseInt(data[0]) + Integer.parseInt(data[2]), Integer.parseInt(data[1]),
                Integer.parseInt(data[0]), Integer.parseInt(data[2]), server_cost);
        boolean node_client = true;
        for (int i = 4; i < graphContent.length; i++) {

            if (graphContent[i].equals("")) {
                node_client = false;
                continue;
            }
            if (node_client) {
                String[] parameter = graphContent[i].split(" ");
                G.addNodeEdge(new Edge(Integer.parseInt(parameter[0]), Integer.parseInt(parameter[1]),
                                Integer.parseInt(parameter[2]), Integer.parseInt(parameter[3]), Integer.parseInt(parameter[2])),
                        true);
            } else {
                String[] parameter = graphContent[i].split(" ");
                int self = Integer.parseInt(parameter[0]);
                int other = Integer.parseInt(parameter[1]);
                int value = 0;

                G.addNodeEdge(new Edge(self + G.getNumberOfNodes(), other,
                        Integer.parseInt(parameter[2]), value, Integer.parseInt(parameter[2])), false);
            }

        }
        return G;
    }

    public static String[] display(String[] graphContent) {
        int par;
        double flag;
        Chromosome best_individual = null;
        Chromosome tmp_individual = null;
        int best_fitness_score = Integer.MAX_VALUE;
        int temp = Integer.parseInt(graphContent[0].split(" ")[0]);
        long deadline;
        if (temp < 200) {
            deadline = 86000;
        } else if(temp<400){
            deadline = 83000;
        }else {
            deadline = 78300;
        }

        Long end;
        Long begin = System.currentTimeMillis();
        Graph G = Deploy.read(graphContent);

        if (G.getNumberOfNodes() > 600) {
            // 高级
            par = 1;
            flag=0.55;
        } else if (G.getNumberOfNodes() > 200) {
            // 中级
            par = 0;
            flag=0.42;
        } else {
            // 低级
            par = -1;
            flag=1;
        }

        if(par==0){

            Optimization opt = new Optimization(G, begin, deadline, graphContent,flag);
            ArrayList<Vertex<Integer>> candidates = opt.influenceMax((int) (G.getNumberOfNodes() * 1));

            opt.setup(candidates);
            // opt.test();

            for (int i = 0; i < 10; i++) {
                opt.init(par);
                tmp_individual = opt.train();
                // System.out.println("第"+i+"次成本："+tmp_individual.fitness_score);

                if (tmp_individual.fitness_score < best_fitness_score) {
                    best_fitness_score = tmp_individual.fitness_score;
                    best_individual = Chromosome.clone(tmp_individual);
                }
                end = System.currentTimeMillis();
                if (end - begin > deadline) {
                    break;
                }
            }


            String[] result = Optimization.getResult(best_individual);
//			System.out.println(best_fitness_score);
            // System.out.println("耗时:"+(System.currentTimeMillis()-begin));

            return result;
        }else{
            Optimization2 opt = new Optimization2(G, begin, deadline, graphContent,flag);
            ArrayList<Vertex<Integer>> candidates = opt.influenceMax((int) (G.getNumberOfNodes() * 1));

            opt.setup(candidates);
            // opt.test();


            for (int i = 0; i < 1; i++) {
                opt.init(par);
                tmp_individual = opt.train();
                // System.out.println("第"+i+"次成本："+tmp_individual.fitness_score);

                if (tmp_individual.fitness_score < best_fitness_score) {
                    best_fitness_score = tmp_individual.fitness_score;
                    best_individual = Chromosome.clone(tmp_individual);
                }
                end = System.currentTimeMillis();
                if (end - begin > deadline) {
                    break;
                }
            }


            String[] result = Optimization2.getResult(best_individual);
//			System.out.println(best_fitness_score);
            // System.out.println("耗时:"+(System.currentTimeMillis()-begin));

            return result;
        }

    }

    public static String stringReverse(String s, Graph g) {
        String result = "";
        String[] ss = s.split(" ");

        // 倒着遍历字符串数组，得到每一个元素
        for (int x = ss.length - 1; x > 0; x--) {
            // 用新字符串把每一个元素拼接起来
            result += ss[x] + " ";
        }
        // 消费节点的id减去网络节点的数目，得到其真实的id
        int bw = Integer.parseInt(ss[0]) - g.getNumberOfNodes();
        result = result + bw;
        return result;
    }
}

/**
 * 边类
 */
class Edge {
    // (v,e)
    final int v;
    final int e;
    final int weight;
    final int value;
    int remain;

    public Edge(int v, int e, int w, int val, int remain) {
        this.weight = w;
        this.value = val;
        this.v = v;
        this.e = e;
        this.remain = remain;
    }

    public int getSelf() {
        return this.v;
    }

    public int getOther() {
        return this.e;
    }

    public int getWeight() {
        return weight;
    }

    public int getValue() {
        return value;
    }

    public int getRemain() {
        return remain;
    }
}

/**
 * 节点类
 */
class Vertex<Item> implements Comparable<Vertex<Item>> {
    // 性价比
    private double performance = 0.0;
    // 每个节点对应一个字符串编码，方便对Vertex类hascode
    private String code;
    // 节点id
    final int id;
    // 该属性用于影响力最大化
    private double priority = 0;
    // 度
    private int degrees = 0;
    // 节点的输出值
    private int outputs = 0;
    // 该节点拥有的边
    private HashMap<Integer, Edge> edges;
    // 该节点的邻居节点
    private LinkedList<Vertex<Item>> neighbor;

    // 该节点是否为普通节点
    private boolean node_flag = false;
    // 该节点是否为服务器
    private boolean client_flag = false;
    // 如果是服务器，则该服务器的需求带宽为demand
    private int demand = 0;
    // 记录被采样概率
    float visited_p = (float) 0.0;

    public Vertex(int id) {
        neighbor = new LinkedList<Vertex<Item>>();
        this.id = id;
        edges = new HashMap<Integer, Edge>();
        this.code = id + "";
    }

    public void sortNeighbors() {
        Collections.sort(neighbor, new Comparator<Vertex<Item>>() {
            public int compare(Vertex<Item> o1, Vertex<Item> o2) {
                if (o1.performance > o2.performance)
                    return -1;
                else if (o1.performance < o2.performance)
                    return 1;
                return 0;
            }
        });
    }

    public void addNeighbors(Vertex<Item> node) {
        neighbor.add(degrees, node);
        degrees += 1;
        if (!this.isClient() && !node.isClient())
            performance += (double) this.getWeight(node) / this.getValue(node);
        if (!this.isClient())
            priority += (double) this.getWeight(node) / this.getValue(node);

    }

    // 如果是消费节点，设置其需求带宽
    public void setDemand(int demand) {
        this.demand = demand;
    }

    // 获得消费节点的需求带宽
    public int getDemand() {
        return demand;
    }

    // 是否为服务器
    public boolean isClient() {
        return client_flag;
    }

    // 是否为普通节点
    public boolean isNode() {
        return node_flag;
    }

    // 设置该节点为消费节点
    public void setClient() {
        client_flag = true;
    }

    // 设置该节点为普通节点
    public void setNode() {
        node_flag = true;
    }

    //
    public String toString() {
        return this.id + "";
    }

    // 获得该节点的所有邻居节点
    public LinkedList<Vertex<Item>> getNeighbors() {
        return neighbor;
    }

    // 获得度
    public int getDegree() {
        return this.degrees;
    }

    // 为该节点添加边
    public void addEdge(Edge edge) {
        outputs += edge.weight;
        this.edges.put(edge.getOther(), edge);
    }

    // 获得指定边上的权重（带宽）
    public int getWeight(int other) {
        return this.edges.get(other).weight;
    }

    public int getWeight(Vertex<Item> other) {
        return getWeight(other.id);
    }

    // 获得指定边上的单价
    public int getValue(int other) {
        return this.edges.get(other).value;
    }

    public int getValue(Vertex<Item> other) {
        return getValue(other.id);
    }

    // 获得该节点的性价比
    public double getPerformance() {
        return performance;
    }

    public int getOutputs() {
        return outputs;
    }

    @Override
    public int hashCode() {
        return code.hashCode();
    }

    public double getPriority() {
        return priority;
    }

    public void setPriority(double dv, int id) {
        Edge edge = edges.get(id);
        double tv = edge.weight / edge.value;

        priority = (dv - 2 * tv - (dv - tv) * tv * 0.01);
    }

    public int compareTo(Vertex<Item> obj) {
        if (!(obj instanceof Vertex))
            new RuntimeException("cuowu");
        Vertex<Item> v = (Vertex<Item>) obj;

        if (this.getPriority() < v.getPriority())
            return 1;
        else if (this.getPriority() == v.getPriority())
            return this.code.compareTo(v.code);
        return -1;
    }
}

/*
 * 步骤 用来存储路径中的每一步 node：该步骤中选择的节点 demand：该步骤提供了需求 value:该步骤单价 cost：该步骤成本
 */
class Step {
    Vertex<Integer> node;
    int demand = 0;
    double value = 0;
    int cost = 0;

    public Step(Vertex<Integer> node, int demand, double value) {
        this.node = node;
        this.demand = demand;
        this.value = value;
    }

    public String toString() {
        return "该节点为" + node + " 承担需求： " + demand + " 价格为:" + value;
    }
}

/*
 * 图 用来构造图 V：节点数量（普通节点+消费节点） num_nodes：普通节点的数量 server_cost：服务器成本 E:边的数量
 * num_client:服务器数量 vertexs：图中所有的节点 Edges:图中所有的边 clients:图中服务器的集合
 *
 */
class Graph {

    private final int V, num_nodes, server_cost;
    int E;
    private int num_client = 0;
    int edge_num = 0;
    private Vertex<Integer>[] vertexs;
    private Edge[] edges;
    private LinkedList<Vertex<Integer>> clients = new LinkedList<Vertex<Integer>>();

    /*
     * 初始化图 给定图中，节点数量，边数量，普通节点数量，服务器数量，服务器成本
     */
    public Graph(int V, int E, int num_nodes, int num_client, int server_cost) {
        this.V = V;
        this.E = E * 2;
        this.num_nodes = num_nodes;
        this.server_cost = server_cost;
        vertexs = (Vertex<Integer>[]) new Vertex[this.V];
        for (int v = 0; v < this.V; v++) {
            vertexs[v] = new Vertex<Integer>(v);
        }
        edges = (Edge[]) new Edge[this.E];
    }

    // 返回服务器成本
    public int getServerCost() {
        return server_cost;
    }

    /*
     * 为给定节点添加边 node_client:用来判断给节点是否为消费节点 True:普通节点 False:消费节点
     */
    public void addNodeEdge(Edge edge, boolean node_client) {
        int v = edge.getSelf();
        int e = edge.getOther();
        if (node_client) {
            vertexs[v].setNode();
            vertexs[e].setNode();
        } else {
            vertexs[v].setClient();
            vertexs[v].setDemand(edge.getWeight());
            clients.add(num_client, vertexs[v]);
            num_client += 1;
        }
        vertexs[v].addEdge(edge);
        vertexs[v].addNeighbors(vertexs[e]);

        Edge other_edge = new Edge(edge.getOther(), edge.getSelf(), edge.getWeight(), edge.getValue(),
                edge.getRemain());
        vertexs[e].addEdge(other_edge);
        vertexs[e].addNeighbors(vertexs[v]);
        if (node_client) {
            edges[edge_num] = edge;
            edge_num += 1;
            edges[edge_num] = other_edge;
            edge_num += 1;
        }
    }

    // 返回图中所有节点
    public Vertex<Integer>[] getVertexs() {
        return vertexs;
    }

    // 返回图中所有消费节点
    public LinkedList<Vertex<Integer>> getClients() {
        return clients;
    }

    // 返回给定节点的度
    public int getNodeDegree(int node) {
        return vertexs[node].getDegree();
    }

    // 返回给定边上的权重（带宽）
    public int getEdgeWeight(int self, int other) {
        return vertexs[self].getWeight(other);
    }

    // 获得给定边上的单价
    public double getEdgeValue(int self, int other) {
        return vertexs[self].getValue(other);
    }

    // 获得给定节点的，邻居节点
    public LinkedList<Vertex<Integer>> getNodeNeighbors(int node) {
        return vertexs[node].getNeighbors();
    }

    public LinkedList<Vertex<Integer>> getNodeNeighbors(Vertex<Integer> node) {
        return getNodeNeighbors(node.id);
    }

    // 获得所有节点的数量
    public int getNumberOfVertexes() {
        return V;
    }

    // 获得边的数量
    public int getNumberOfEdges() {
        return E;
    }

    // 给定节点的id，获得该节点类
    public Vertex<Integer> getNode(int id) {
        return vertexs[id];
    }

    // 获得普通节点的数量
    public int getNumberOfNodes() {
        return this.num_nodes;
    }

    // 获得消费节点的数量
    public int getNumberOfClient() {
        return this.num_client;
    }
}

/*
 * 染色体（解） boolean[] gen:存储服务器组合方式 例： gen[1] = {true,false,false,true,false}
 * 则，候选一号和四号节点为服务器，其他为普通节点。
 *
 * fitness_score:该解得适应度值 size:gen的大小。其与candidate（候选集合）集合大小保持一致
 */
class Chromosome {
    boolean[] gene;
    int fitness_score = 0;
    int size = 0;

    public Chromosome() {
    }

    public Chromosome(int size) {
        if (size <= 0) {
            return;
        }
        initGeneSize(size);
        for (int i = 0; i < size; i++) {
            gene[i] = Math.random() >= 0.5;
        }
    }

    public Chromosome(int size, double[] performance, LinkedList<Vertex<Integer>> clients,
                      ArrayList<Vertex<Integer>> candidates, float rate) {
        if (size <= 0) {
            return;
        }

        initGeneSize(size);
        int len = (int) (size * 0.6);
        int idx;
        Random random = new Random();
        for (Vertex<Integer> client : clients) {
            idx = client.getNeighbors().getFirst().id;
            if (performance[idx] > Math.random()) {
                gene[idx] = true;
            } else {
                gene[idx] = false;
                idx = candidates.get(random.nextInt(len)).id;
                gene[idx] = performance[idx] > Math.random();
            }
        }
    }

    public Chromosome(int size, double[] performance, LinkedList<Vertex<Integer>> clients,
                      ArrayList<Vertex<Integer>> candidates, float rate, Chromosome sample,
                      LinkedList<Integer> certain_servers) {
        if (size <= 0) {
            return;
        }
        initGeneSize(size);
        Random random = new Random();
        int len = (int) (candidates.size() * rate);
        int idx;
		/*
		for (int i = 0; i < sample.gene.length; i++) {
		   if (sample.gene[i]) {
		       this.gene[i] = Math.random()>0.5;
		   }
		}*/
        for(int i : certain_servers){
            this.gene[i] = Math.random()>0.5;
        }
    }
    public Chromosome(int size, double[] performance, LinkedList<Vertex<Integer>> clients,
                      ArrayList<Vertex<Integer>> candidates, float rate, Chromosome sample) {
        if (size <= 0) {
            return;
        }
        initGeneSize(size);
        Random random = new Random();
        int len = (int) (candidates.size() * rate);
        int idx;
        for (int i = 0; i < sample.gene.length; i++) {
            if (sample.gene[i]) {
                if (Math.random() > 0.5)
                    this.gene[i] = true;
                else {
                    this.gene[i] = false;
                    idx = candidates.get(random.nextInt(len)).id;
                    this.gene[idx] = performance[idx] > Math.random();
                }
            }
        }
    }

    void initGeneSize(int size) {
        if (size <= 0) {
            return;
        }
        this.size = size;
        gene = new boolean[size];
    }

    public boolean[] getGene() {
        return gene;
    }

    // 对染色体克隆
    public static Chromosome clone(final Chromosome target) {
        if (target == null || target.gene == null)
            return null;
        Chromosome result = new Chromosome();
        result.initGeneSize(target.size);
        for (int i = 0; i < target.size; i++) {
            result.gene[i] = target.gene[i];
        }
        result.fitness_score = target.fitness_score;
        return result;
    }
}

/*
 * 优化寻找服务器
 */
class Optimization {

    /*
     * 种群基本信息配置 best_individual:最优个体 global_best_fitness：最有适应度值 gen_size：染色体大小
     * pop_size:种群大小 max_gen:最大迭代次数 mutation_rate:变异概率 candidates：候选种子集
     * population:种群
     */
    private Chromosome sample_individual;
    static String[] graphContent;
    private Chromosome best_individual;
    private int global_best_fitness = Integer.MAX_VALUE;
    private int gen_size;
    private int pop_size = 400;
    private final int max_gen = 100000;
    private final int mate_size = (int) (pop_size * 0.25);
    private final double mutation_rate = 1;
    static private ArrayList<Vertex<Integer>> candidates;
    private ArrayList<Chromosome> population;
    private int end_condition;
    /*
     * performance_init：性价比 outputs_init:每个节点的输出带宽 weights_init：两个边上的带宽
     * clients_init:消费节点
     */
    static private double max_performance = 0;
    static private double min_performance = Integer.MAX_VALUE;
    static private double[] normalization_performance;

    static private int total_visited_num=0;
    static private int[] visited_num;
    static double[] visited_p;
    static private int[][] values_init;
    static private double[] performance_init;
    static private int[] outputs_init;
    static private int[][] weights_init;
    static private LinkedList<Vertex<Integer>> clients_init;
    static public ArrayList<Vertex<Integer>> tempNode;
    static Graph G;
    public static double flag;
    // 计时器
    long start_time;
    long deadline;

    //
    public Optimization(Graph G, long start_time, long deadline, String[] graphContent,double flag) {
        this.deadline = deadline;
        this.start_time = start_time;
        Optimization.G = G;
        Optimization.graphContent = graphContent;
        this.flag=flag;
    }

    // 测试
    public void test(ArrayList<Vertex<Integer>> candidates) {
        Optimization.candidates = candidates;
        this.gen_size = G.getNumberOfNodes();
        visited_num = new int[G.getNumberOfNodes()];
        visited_p = new double[G.getNumberOfNodes()];
        best_individual = null;
        normalization_performance = new double[G.getNumberOfNodes()];
        performance_init = new double[G.getNumberOfVertexes()];
        outputs_init = new int[G.getNumberOfVertexes()];
        weights_init = new int[G.getNumberOfVertexes()][G.getNumberOfVertexes()];
        clients_init = G.getClients();
        values_init = new int[G.getNumberOfVertexes()][G.getNumberOfVertexes()];

        // 对消费者排序
        Collections.sort(clients_init, new Comparator<Vertex<Integer>>() {
            public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
                if (o1.getDemand() > o2.getDemand())
                    return -1;
                else if (o1.getDemand() < o2.getDemand())
                    return 1;
                else
                    return 0;
            }
        });
        //
        for (int i = 0; i < G.getNumberOfVertexes(); i++) {
            Vertex<Integer> node = G.getNode(i);
            performance_init[i] = node.getPerformance();
            outputs_init[i] = node.getOutputs();
            LinkedList<Vertex<Integer>> neighbors = node.getNeighbors();
            for (Vertex<Integer> neighbor : neighbors) {
                if (neighbor.isNode()) {
                    values_init[i][neighbor.id] = node.getValue(neighbor);
                    weights_init[i][neighbor.id] = node.getWeight(neighbor);
                }
                if (node.isClient())
                    weights_init[node.id][neighbor.id] = node.getDemand();
            }
            if (node.isNode()) {
                node.sortNeighbors();
                if (performance_init[i] > max_performance) {
                    max_performance = performance_init[i];
                } else if (performance_init[i] < min_performance) {
                    min_performance = performance_init[i];
                }
            }
        }
        // 归一化
        for (int i = 0; i < normalization_performance.length; i++) {
            normalization_performance[i] = (performance_init[i] - min_performance)
                    / (max_performance - min_performance);
        }
    }

    public void setup(ArrayList<Vertex<Integer>> candidates) {
        int min_cost = Integer.MAX_VALUE;
        int tmp_cost = 0;
        Optimization.candidates = candidates;
        this.gen_size = G.getNumberOfNodes();
        visited_num = new int[G.getNumberOfNodes()];
        visited_p = new double[G.getNumberOfNodes()];
        best_individual = null;
        normalization_performance = new double[G.getNumberOfNodes()];
        performance_init = new double[G.getNumberOfVertexes()];
        outputs_init = new int[G.getNumberOfVertexes()];
        weights_init = new int[G.getNumberOfVertexes()][G.getNumberOfVertexes()];
        clients_init = G.getClients();

        // 对消费者排序
        Collections.sort(clients_init, new Comparator<Vertex<Integer>>() {
            public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
                if (o1.getDemand() > o2.getDemand())
                    return -1;
                else if (o1.getDemand() < o2.getDemand())
                    return 1;
                else
                    return 0;
            }
        });
        //
        for (int i = 0; i < G.getNumberOfVertexes(); i++) {
            Vertex<Integer> node = G.getNode(i);
            performance_init[i] = node.getPerformance();
            outputs_init[i] = node.getOutputs();
            LinkedList<Vertex<Integer>> neighbors = node.getNeighbors();
            for (Vertex<Integer> neighbor : neighbors) {
                if (neighbor.isNode())
                    weights_init[i][neighbor.id] = node.getWeight(neighbor);
                if (node.isClient())
                    weights_init[node.id][neighbor.id] = node.getDemand();
            }
            if (node.isNode()) {
                node.sortNeighbors();
                if (performance_init[i] > max_performance) {
                    max_performance = performance_init[i];
                } else if (performance_init[i] < min_performance) {
                    min_performance = performance_init[i];
                }
            }
        }
        // 归一化
        for (int i = 0; i < normalization_performance.length; i++) {
            normalization_performance[i] = (performance_init[i] - min_performance)
                    / (max_performance - min_performance);
        }
        //
        sample_individual = new Chromosome();
        sample_individual.initGeneSize(gen_size);

        int idx;

        LinkedList<Vertex<Integer>> set = new LinkedList<Vertex<Integer>>();
//		for (Vertex<Integer> client : clients_init) {
//		System.out.println(clients_init.size()*flag);
        for (int i=0;i< clients_init.size()*flag;i++) {
//			idx = client.getNeighbors().getFirst().id;
//			sample_individual.gene[idx] = true;
//			set.add(client.getNeighbors().getFirst());

            idx = clients_init.get(i).getNeighbors().getFirst().id;
            sample_individual.gene[idx] = true;
            set.add(clients_init.get(i).getNeighbors().getFirst());

        }
//		for(int i=0;i<sample_individual.gene.length*0.5;i++){
//
//			sample_individual.gene[candidates.get(i).id]=true;
//			set.add(candidates.get(i));
//		}


//		// 此处基于performance与priotiry相同
//		Collections.sort(set, new Comparator<Vertex<Integer>>() {
//			public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
//				if (o1.getPerformance() > o2.getPerformance())
//					return 1;
//				else if (o1.getPerformance() < o2.getPerformance())
//					return -1;
//				return 0;
//			}
//		});
//		for (int i = 0; i < set.size(); i++) {
//			sample_individual.gene[set.get(i).id] = false;
//			tmp_cost = getCostBk(sample_individual);
//			if (tmp_cost < min_cost) {
//				min_cost = tmp_cost;
//			} else {
//				sample_individual.gene[set.get(i).id] = true;
//			}
//		}
//		// 下面针对sample进行第二次优化，因为后面穷尽所有情况，所以没有区别
//		for (int i = 0; i < sample_individual.gene.length; i++) {
//			if (sample_individual.gene[i]) {
//				sample_individual.gene[i] = false;
//				tmp_cost = getCostBk(sample_individual);
//				if (tmp_cost < min_cost) {
//					min_cost = tmp_cost;
//				} else {
//					sample_individual.gene[i] = true;
//				}
//			}
//		}


//		for(int i=0;i<G.getNumberOfNodes();i++){
//			int index = candidates.get(i).id;
//			sample_individual.gene[index] = true;
//			tmp_cost = getCostBk(sample_individual);
//			if(tmp_cost<=min_cost){
//				if(tmp_cost < min_cost){
//					min_cost=tmp_cost;
//					System.out.println("-----------------");
//				}
//			}else{
//				sample_individual.gene[index]=false;
//			}
//		}
//		for(int j=0;j<100;j++){
//			for (int i = 0; i < sample_individual.gene.length; i++) {
//				if (sample_individual.gene[i]) {
//					sample_individual.gene[i] = false;
//					sample_individual.gene[j] = false;
//					tmp_cost = getCostBk(sample_individual);
//					if (tmp_cost < min_cost) {
//						min_cost = tmp_cost;
//					} else {
//						sample_individual.gene[i] = true;
//					}
//				}
//			}
//		}
//



    }

    // 种群初始化
    public void init(int par) {

        int min_cost = Integer.MAX_VALUE;
        int tmp_cost = 0;
        best_individual = null;
        // 初始化种群
        population = new ArrayList<Chromosome>();
        min_cost = Integer.MAX_VALUE;
        tmp_cost = 0;

        if (par == 1) {
            pop_size = 5;
            end_condition = 50000;
        } else if (par == 0) {
            pop_size = 5;
            end_condition = 50000;
        } else {
            pop_size = 200;
            end_condition = 50000;
        }

        for (int i = 0; i < pop_size; i++) {
            Chromosome individual = null;
            if (i < (int) (pop_size * 1.0)) {
                individual = new Chromosome();
                individual.initGeneSize(gen_size);
                // System.out.println(sample_individual.gene.length);
                for (int j = 0; j < sample_individual.gene.length; j++) {
                    individual.gene[j] = sample_individual.gene[j];
                }
            }
//			 else
//				 if(i < (int)(pop_size*0.0)){
//			 individual = new Chromosome(gen_size,
//			 normalization_performance,
//			 clients_init,
//			 candidates,
//			 (float) 0.6,
//			 sample_individual);
//			 }else {
//			 individual = new Chromosome(gen_size,
//			 normalization_performance,
//			 clients_init,
//			 candidates,
//			 (float)1.0);
//			 }
            tmp_cost = getCost(individual);
            // System.out.println(tmp_cost);
            individual.fitness_score = tmp_cost;
            population.add(individual);
            if (min_cost > tmp_cost) {
                min_cost = tmp_cost;
                best_individual = Chromosome.clone(individual);
                best_individual.fitness_score = min_cost;
                global_best_fitness = min_cost;
            }
        }
        // System.out.println(global_best_fitness+"
        // "+best_individual.fitness_score);
    }

    // 选择
    public ArrayList<Chromosome> selection() {
        ArrayList<Chromosome> next_population = new ArrayList<Chromosome>();
        next_population.add(best_individual);
        while (next_population.size() < mate_size) {
            Random rand = new Random();
            int i = rand.nextInt(pop_size);
            int j = rand.nextInt(pop_size);
            if (population.get(i).fitness_score < population.get(j).fitness_score)
                next_population.add(Chromosome.clone(population.get(i)));
            else
                next_population.add(Chromosome.clone(population.get(j)));
        }
        return next_population;
    }

    // 交叉
    public void crossover(ArrayList<Chromosome> next_population) {
        Chromosome a;
        Chromosome b;
        int index;
        Chromosome a_next;
        Chromosome b_next;
        while (next_population.size() < pop_size) {
            Random random = new Random();
            a = next_population.get(random.nextInt(mate_size));
            b = next_population.get(random.nextInt(mate_size));
//			ArrayList<> a=new ArrayList
            index = random.nextInt(gen_size);
            a_next = new Chromosome();
            b_next = new Chromosome();
            a_next.initGeneSize(gen_size);
            b_next.initGeneSize(gen_size);
            for (int i = 0; i < index; i++) {
                a_next.gene[i] = a.gene[i];
                b_next.gene[i] = b.gene[i];
            }
            for (int i = index; i < gen_size; i++) {
                a_next.gene[i] = b.gene[i];
                b_next.gene[i] = a.gene[i];
            }
            next_population.add(a_next);
            next_population.add(b_next);
        }
    }

    // 变异
    public void mutate(ArrayList<Chromosome> next_population) {
        Chromosome individual;
        Random random = new Random();
        Vertex<Integer> node;
        int idx;
        for (int i = 1; i < next_population.size(); i++) {
            individual = next_population.get(i);
            if (Math.random() < mutation_rate) {
                for (int t = 0; t < 1; t++) {
                    node = candidates.get(random.nextInt(candidates.size()));
                    idx = node.id;
                    individual.gene[idx] = !individual.gene[idx];// = 0.5 >
                    // Math.random();
                }
            }
        }
    }


    // 局部搜索
    public void localSearch() {
        Chromosome tmp = Chromosome.clone(best_individual);
        boolean flag = true;
        Random random;
        int idx;
        int min_cost = global_best_fitness;
        int tmp_cost = 0;
        while (flag) {
            random = new Random();
            for (int i = 0; i < 7; i++) {
                idx = random.nextInt(gen_size);
                if (normalization_performance[idx] > Math.random())
                    tmp.gene[idx] = true;
                else
                    tmp.gene[idx] = false;
            }
            tmp_cost = getCost(tmp);
            if (tmp_cost < min_cost) {
                min_cost = tmp_cost;
                tmp.fitness_score = tmp_cost;
                best_individual = Chromosome.clone(tmp);
                best_individual.fitness_score = min_cost;
                global_best_fitness = min_cost;
            } else {
                flag = false;
            }
        }
    }



    // 获得下一代
    public ArrayList<Chromosome> genNextPopulation() {
        ArrayList<Chromosome> next_population = selection();
        crossover(next_population);
        mutate(next_population);
        localSearch();
        return next_population;
    }

    // 训练
    public Chromosome train() {
        int count = 0;
        int last_min_cost = 0;
        int min_cost = Integer.MAX_VALUE;
        int tmp_cost;
        for (int t = 0; t < max_gen; t++) {
            tmp_cost = 0;
            population = genNextPopulation();

            for (Chromosome individual : population) {
                tmp_cost = getCost(individual);
                individual.fitness_score = tmp_cost;
                if (min_cost > tmp_cost) {
                    min_cost = tmp_cost;
                    best_individual = Chromosome.clone(individual);
                    best_individual.fitness_score = min_cost;
                    global_best_fitness = min_cost;
                    localSearch();
                }
            }

            // 判断是否收敛
            if (last_min_cost == global_best_fitness) {
                count += 1;
                if (count == end_condition) {
                    return best_individual;
                }
            } else {
                count = 0;
            }

            last_min_cost = min_cost;
            Long end_time = System.currentTimeMillis();
            // 判断是否超时
            if (end_time - start_time > deadline) {
                return best_individual;
            }
//			System.out.println(t + "   " + global_best_fitness);
        }
        return best_individual;
    }

    // 获得结果
    public static String[] getResult(Chromosome individual) {
        boolean[] gen = individual.getGene();
        ArrayList<Integer> server_set = new ArrayList<Integer>();
        for (int i = 0; i < gen.length; i++) {
            if (gen[i])
                server_set.add(G.getNode(i).id);///// 获取最优服务器位置
        }
        String[] result;
		
		result = ZKW.getWalk(graphContent, server_set, weights_init,clients_init);

        // if(flag<0.4){
            // result = ZKW.getWalk(graphContent, ZKW.methodOfzkw(graphContent, ZKW.methodOfzkw(graphContent, server_set)), weights_init,clients_init);
        // }else{
            // result = ZKW.getWalk(graphContent, ZKW.methodOfzkw(graphContent, server_set), weights_init,clients_init);
        // }

        return result;
        // result= Deploy.methodOfFlow(graphContent,
        // ZKW.methodOfzkw(graphContent, server_set));
        // return result;
    }

    // 目标函数，获得每个解得成本
    public int getCost(Chromosome individual) {
        int result = 0;
        boolean[] gen = individual.getGene();
        // server_set：服务器集合
        HashSet<Vertex<Integer>> server_set = new HashSet<Vertex<Integer>>();
        HashSet<Integer> sets = new HashSet<Integer>();
        for (int idx = 0; idx < gen.length; idx++) {
            if (gen[idx])
                server_set.add(G.getNode(idx));
        }
        // 寻找该解下的所有路径，同时对不满足需求的解添加服务器
        LinkedList<LinkedList<Step>> walks = findWalks(G, server_set, clients_init, performance_init, outputs_init,
                weights_init, G.getNumberOfVertexes());


        //最大流获取成本
//		 ArrayList<Integer> servers=new ArrayList<Integer>();
//		 for(Vertex<Integer> i:server_set){
//			 servers.add(i.id);
//		 }
//		 result= ZKW.methodOfzkw1(graphContent, servers);
//		 if(result==-1){
//			 result=Integer.MAX_VALUE;
//		 }
//		 for(Integer i : servers){
//			 individual.gene[i] = true;
//		 }

//
//		// 计算成本------------启发式搜索路径计算成本
        individual.gene = new boolean[individual.size];
        for (LinkedList<Step> walk : walks) {
            for (Step step : walk) {
                // 确保当路径到达新增服务器时，停止计算成本
				/*
				 * 例如： 0->1->2->5->4 其中，2号节点在之后被选择为服务器。则计算成本是，应当到达2号节点，停止。
				 */
                if (server_set.contains(step.node)) {
                    sets.add(step.node.id);
                    result += step.cost;
                    total_visited_num += 1;
                    break;
                }
            }
        }
        for (Integer i : sets) {
            visited_num[i] += 1;
            individual.gene[i] = true;
        }
        result += G.getServerCost() * sets.size();



        return result;
    }

    public int getCostBk(Chromosome individual) {
        int result = 0;
        boolean[] gen = individual.getGene();
        // server_set：服务器集合
        HashSet<Vertex<Integer>> server_set = new HashSet<Vertex<Integer>>();
        HashSet<Integer> sets = new HashSet<Integer>();
        for (int idx = 0; idx < gen.length; idx++) {
            if (gen[idx])
                server_set.add(G.getNode(idx));
        }
        // 寻找该解下的所有路径
        LinkedList<LinkedList<Step>> walks = findWalks(G, server_set, clients_init, performance_init, outputs_init,
                weights_init, G.getNumberOfVertexes());

        for (LinkedList<Step> walk : walks) {
            for (Step step : walk) {
                // 确保当路径到达新增服务器时，停止计算成本
				/*
				 * 例如： 0->1->2->5->4 其中，2号节点在之后被选择为服务器。则计算成本是，应当到达2号节点，停止。
				 */
                if (server_set.contains(step.node)) {
                    sets.add(step.node.id);
                    result += step.cost;
                    break;
                }
            }
        }
        result += G.getServerCost() * sets.size();
        return result;
    }

    // 寻找所有路径
    public static LinkedList<LinkedList<Step>> findWalks(Graph G, HashSet<Vertex<Integer>> server_set,
                                                         LinkedList<Vertex<Integer>> clients, double[] performance, int[] outputs, int[][] weights, int len) {

        LinkedList<LinkedList<Step>> walks = new LinkedList<LinkedList<Step>>();
        // outputs_neighbor_server，该节点的邻居节点有服务器，计算其与服务器的带宽和
        // neighbor_has_server：判断该节点的邻居节点是否有服务器
        boolean[] neighbor_has_server = new boolean[len];
        int[] outputs_neighbor_server = new int[len];
        // 克隆信息
        double[] performance_tmp = performance;
        int[] outputs_tmp = outputs.clone();
        int[][] weights_tmp = new int[len][];
        for (int i = 0; i < len; i++) {
            weights_tmp[i] = weights[i].clone();
        }
        // 根据服务器，更新网络中各个节点的性价比
        for (Vertex<Integer> server : server_set) {
            for (Vertex<Integer> neighbor : server.getNeighbors()) {
                if (neighbor.isNode() && !server_set.contains(neighbor)) {
                    neighbor_has_server[neighbor.id] = true;
                    outputs_neighbor_server[neighbor.id] += neighbor.getWeight(server);
                }
            }
        }
        // 为每一个服务器，寻找路径
        for (Vertex<Integer> client : clients) {
            int demand = client.getDemand();
            int remain_demand = demand;
            // 确保满足每一个服务器的需求带宽
            while (remain_demand != 0) {
                LinkedList<Step> walk = find(client, client.getDemand(), server_set, neighbor_has_server,
                        performance_tmp, weights_tmp, outputs_neighbor_server, outputs_tmp);

                walks.addLast(walk);
                remain_demand -= walk.getLast().demand;
            }
        }

        return walks;
    }

    // 给定服务器和需要承担的带宽，寻找路径
    public static LinkedList<Step> find(Vertex<Integer> client, int demand, final HashSet<Vertex<Integer>> server_set,
                                        final boolean[] neighbor_has_server, final double[] performance, int[][] weights,
                                        int[] outputs_neighbor_server, int[] outputs) {

        HashSet<Vertex<Integer>> selected_nodes = new HashSet<Vertex<Integer>>();
        int cost = 0;
        LinkedList<Step> walk = new LinkedList<Step>();
        walk.addLast(new Step(client, demand, 0));

        // 用于加快排序的参数
        int idx_0 = 0;
        int idx_1 = 0;
        boolean flag_sort = true;
        Vertex<Integer> tmp_node;
        // 用于控制路径寻找深度
        int count = 0;
        final int max_find_count = 6;

        boolean flag = true;
        while (flag) {
            Step current_step = walk.getLast();
            final Vertex<Integer> current_node = current_step.node;
            int current_demand = current_step.demand;
            // 生成新的邻居节点集
            LinkedList<Vertex<Integer>> raw_neighbors = current_node.getNeighbors();
            LinkedList<Vertex<Integer>> neighbors = new LinkedList<Vertex<Integer>>();

            for (Vertex<Integer> neighbor : raw_neighbors) {
                if (neighbor.isNode() && weights[current_node.id][neighbor.id] > 0
                        && !selected_nodes.contains(neighbor)) {

                    if (server_set.contains(neighbor)) {
                        for (int i = 0; i < neighbors.size(); i++) {
                            tmp_node = neighbors.get(i);
                            if (server_set.contains(tmp_node)) {
                                if (neighbor.getValue(current_node) < tmp_node.getValue(current_node)) {
                                    neighbors.add(i, neighbor);
                                    flag_sort = false;
                                    break;
                                }
                            } else {
                                neighbors.add(i, neighbor);
                                flag_sort = false;
                                break;
                            }
                        }
                        if (flag_sort)
                            neighbors.addLast(neighbor);
                        flag_sort = true;
                        idx_0 += 1;
                        idx_1 += 1;
                    } else if (neighbor_has_server[neighbor.id]) {
                        for (int i = idx_0; i < neighbors.size(); i++) {
                            tmp_node = neighbors.get(i);
                            if (server_set.contains(tmp_node)) {
                                ;
                            } else if (neighbor_has_server[tmp_node.id]) {
                                if (neighbor.getValue(current_node) < tmp_node.getValue(current_node)) {
                                    neighbors.add(i, neighbor);
                                    flag_sort = false;
                                    break;
                                }
                            } else {
                                neighbors.add(i, neighbor);
                                flag_sort = false;
                                break;
                            }
                        }
                        if (flag_sort)
                            neighbors.addLast(neighbor);
                        idx_1 += 1;
                        flag_sort = true;
                    } else {
                        for (int i = idx_1; i < neighbors.size(); i++) {
                            tmp_node = neighbors.get(i);
                            if (server_set.contains(tmp_node)) {
                                ;
                            } else if (neighbor_has_server[tmp_node.id]) {
                                ;
                            } else {
                                if (neighbor.getValue(current_node) < tmp_node.getValue(current_node)) {
                                    neighbors.add(i, neighbor);
                                    flag_sort = false;
                                    break;
                                }
                            }
                        }
                        if (flag_sort)
                            neighbors.addLast(neighbor);
                        flag_sort = true;
                    }
                }
            }
			/*
			 * System.out.println("开始"); System.out.println(server_set);
			 * System.out.println(selected_nodes);
			 * System.out.println(raw_neighbors);
			 * System.out.println(current_node); System.out.println("测试一");
			 * for(Vertex<Integer> neighbor : neighbors){
			 * System.out.println(neighbor+" "+neighbor.getPerformance()+"  "
			 * +neighbor.getValue(current_node)); }
			 */
            // 根据规则，排序
			/*
			 * neighbors = new LinkedList<Vertex<Integer>>(); for
			 * (Vertex<Integer> neighbor : raw_neighbors) { if
			 * (neighbor.isNode() && weights[current_node.id][neighbor.id] > 0
			 * && !selected_nodes.contains(neighbor)){
			 * neighbors.addLast(neighbor); } } Collections.sort(neighbors, new
			 * Comparator<Vertex<Integer>>() { public int
			 * compare(Vertex<Integer> o1, Vertex<Integer> o2) {
			 * if(server_set.contains(o1) && !server_set.contains(o2)) return
			 * -1; else if((!server_set.contains(o1) &&
			 * server_set.contains(o2))) return 1;
			 *
			 * if ((o1.isNode() && neighbor_has_server[o1.id] && o2.isNode() &&
			 * !neighbor_has_server[o2.id]) ) return -1; else if((o1.isNode() &&
			 * !neighbor_has_server[o1.id] && o2.isNode() &&
			 * neighbor_has_server[o2.id])) return 1;
			 *
			 * if(o1.getValue(current_node) > o2.getValue(current_node)) return
			 * 1; if(o1.getValue(current_node) < o2.getValue(current_node))
			 * return -1;
			 *
			 * if(performance[o1.id] > performance[o2.id]) return -1; else
			 * if(performance[o1.id] < performance[o2.id]) return 1; return 0; }
			 * });
			 */
			/*
			 * //System.out.println("测试二"); //for(Vertex<Integer> neighbor :
			 * neighbors){ //
			 * System.out.println(neighbor+" "+neighbor.getPerformance()+"  "
			 * +neighbor.getValue(current_node)); //}
			 */
            // 寻找下一步
            Step next_step = findNextStep(current_node, current_demand, neighbors, server_set, neighbor_has_server,
                    weights, outputs_neighbor_server, performance, outputs);
            // 判断你是否停止寻找节点，终止搜索。并且计算每一步的成本。
            // 在终止前，应该更新网络属性。例如，outputs，weights和perforamnce
            if (next_step != null) {
                walk.addLast(next_step);
                selected_nodes.add(next_step.node);
                count += 1;
                if (server_set.contains(next_step.node)) {
                    flag = false;
                    int size = walk.size();
                    Vertex<Integer> last = client;
                    for (int i = 1; i < size; i++) {
                        Step step = walk.get(i);
                        if (i > 1) {
                            cost += walk.getLast().demand * step.value;
                            step.cost = cost;
                        }
                        weights[last.id][step.node.id] -= walk.getLast().demand;
                        if (neighbor_has_server[step.node.id])
                            outputs_neighbor_server[step.node.id] -= walk.getLast().demand;
                        outputs[step.node.id] -= walk.getLast().demand;
                        last = step.node;
                    }
                } else if (count > max_find_count) {
                    flag = false;
                    int size = walk.size();
                    Vertex<Integer> last = client;
                    for (int i = 1; i < size; i++) {
                        Step step = walk.get(i);
                        if (i > 1) {
                            cost += walk.getLast().demand * step.value;
                            step.cost = cost;
                        }
                        weights[last.id][step.node.id] -= walk.getLast().demand;
                        if (neighbor_has_server[step.node.id])
                            outputs_neighbor_server[step.node.id] -= walk.getLast().demand;
                        outputs[step.node.id] -= walk.getLast().demand;
                        last = step.node;
                    }
                    // 更新，新增服务器
                    Vertex<Integer> new_server = walk.getLast().node;
                    server_set.add(new_server);
                    // 更新新增服务器的邻居节点的配置信息，如perforamnce，outputs_neighbor_server
                    for (Vertex<Integer> neighbor : new_server.getNeighbors()) {
                        if (neighbor.isNode() && !server_set.contains(neighbor)) {
                            neighbor_has_server[neighbor.id] = true;
                            outputs_neighbor_server[neighbor.id] += weights[neighbor.id][new_server.id];
                        }
                    }
                }
            } else {
                flag = false;
                int size = walk.size();
                Vertex<Integer> last = client;
                for (int i = 1; i < size; i++) {
                    Step step = walk.get(i);
                    if (i > 1) {
                        cost += walk.getLast().demand * step.value;
                        step.cost = cost;
                    }
                    weights[last.id][step.node.id] -= walk.getLast().demand;
                    if (neighbor_has_server[step.node.id])
                        outputs_neighbor_server[step.node.id] -= walk.getLast().demand;
                    outputs[step.node.id] -= walk.getLast().demand;
                    last = step.node;
                }
                // 更新，新增服务器
                Vertex<Integer> new_server = walk.getLast().node;
                server_set.add(new_server);
                // 更新新增服务器的邻居节点的配置信息，如perforamnce，outputs_neighbor_server
                for (Vertex<Integer> neighbor : new_server.getNeighbors()) {
                    if (neighbor.isNode() && !server_set.contains(neighbor)) {
                        neighbor_has_server[neighbor.id] = true;
                        outputs_neighbor_server[neighbor.id] += weights[neighbor.id][new_server.id];
                    }
                }
            }
        }
        return walk;
    }

    // 寻找一条路径中的下一步骤
    public static Step findNextStep(Vertex<Integer> current_node, int current_demand,
                                    LinkedList<Vertex<Integer>> neighbors, final HashSet<Vertex<Integer>> server_set,
                                    boolean[] neighbor_has_server, int[][] weights, int[] outputs_neighbor_server, double[] performance,
                                    int[] outputs) {

        if (neighbors.isEmpty())
            return null;
        Vertex<Integer> neighbor = neighbors.removeFirst();

        if (server_set.contains(neighbor) && weights[current_node.id][neighbor.id] > 0) {
            int support = Math.min(current_demand, weights[current_node.id][neighbor.id]);
            return new Step(neighbor, support, current_node.getValue(neighbor));
        } else if (neighbor_has_server[neighbor.id] && outputs_neighbor_server[neighbor.id] > 0) {
            int server_support = Math.min(outputs_neighbor_server[neighbor.id], weights[current_node.id][neighbor.id]);
            int support = Math.min(current_demand, server_support);

            return new Step(neighbor, support, current_node.getValue(neighbor));
        } else if (performance[neighbor.id] > performance[current_node.id]
                && weights[current_node.id][neighbor.id] > 0) {
            int node_support = Math.min(outputs[neighbor.id], weights[current_node.id][neighbor.id]);
            int support = Math.min(current_demand, node_support);
            return new Step(neighbor, support, current_node.getValue(neighbor));
        }
        return findNextStep(current_node, current_demand, neighbors, server_set, neighbor_has_server, weights,
                outputs_neighbor_server, performance, outputs);
    }

    // 影响力最大化，选择候选集

    public ArrayList<Vertex<Integer>> influenceMax(int keyNum) {

        ArrayList<Vertex<Integer>> maxVertexs = new ArrayList<Vertex<Integer>>();

        TreeSet<Vertex> sortVertexsByPriority = new TreeSet<Vertex>();
        // 此处是原方法的基于度的节点的存储方式
        ArrayList<Integer> vertexsOfNotClient = new ArrayList<Integer>();

        Vertex<Integer>[] vertexs = this.G.getVertexs();
        LinkedList<Vertex<Integer>> clients = this.G.getClients();

        for (Vertex v : vertexs) {
            if (clients.contains(v)) {
                continue;
            }
            sortVertexsByPriority.add(v);
            vertexsOfNotClient.add(v.id);
        }

        for (int j = 0; j < keyNum; j++) {
            Vertex v = sortVertexsByPriority.pollFirst();
            maxVertexs.add(v);
            LinkedList<Vertex> neighbor = v.getNeighbors();
            int value = 0;
            for (Vertex vn : neighbor) {
                if (!maxVertexs.contains(vn) && !clients.contains(vn)) {
                    sortVertexsByPriority.remove(vn);
                    vn.setPriority(vn.getPriority(), v.id);
                    sortVertexsByPriority.add(vn);
                }
            }

        }

        return maxVertexs;
    }

    public ArrayList<Vertex<Integer>> influenceMax() {
        ArrayList<Vertex<Integer>> candidates = new ArrayList<Vertex<Integer>>();
        int[][] value_init = new int[G.getNumberOfVertexes()][G.getNumberOfVertexes()];
        int[] visited_num_tmp = new int[G.getNumberOfVertexes()];
        int[] values = new int[G.getNumberOfVertexes()];
        LinkedList<Vertex<Integer>> neighbors;
        LinkedList<Vertex<Integer>> current_neighbors;
        LinkedList<Vertex<Integer>> next_neighbors;

        Vertex<Integer> node;
        for (int i = 0; i < G.getNumberOfNodes(); i++) {
            node = G.getNode(i);
            neighbors = node.getNeighbors();
            for (Vertex<Integer> neighbor : neighbors) {
                value_init[node.id][neighbor.id] = (int) node.getValue(neighbor);
            }
        }
        //
        for (Vertex<Integer> client : G.getClients()) {
            for (int idx = 0; idx < values.length; idx++) {
                values[idx] = Integer.MAX_VALUE;
            }
            current_neighbors = new LinkedList<Vertex<Integer>>();
            values[client.id] = 0;
            current_neighbors.add(client);
            for (int i = 0; i < 3; i++) {
                next_neighbors = new LinkedList<Vertex<Integer>>();
                while (!current_neighbors.isEmpty()) {
                    node = current_neighbors.removeFirst();
                    neighbors = node.getNeighbors();
                    for (Vertex<Integer> neighbor : neighbors) {

                        visited_num_tmp[neighbor.id] += 1;
                        next_neighbors.add(neighbor);
                    }
                }
                current_neighbors = next_neighbors;
            }
        }
        int max = 0;
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < G.getNumberOfNodes(); i++) {
            if (visited_num_tmp[i] != 0) {
                if (visited_num_tmp[i] > max) {
                    max = visited_num_tmp[i];
                }
                if (visited_num_tmp[i] < min) {
                    min = visited_num_tmp[i];
                }
            }
        }
        for (int i = 0; i < G.getNumberOfNodes(); i++) {
            G.getNode(i).visited_p = ((visited_num_tmp[i] - min) / (float) (max - min));
            candidates.add(G.getNode(i));

        }
        Collections.sort(candidates, new Comparator<Vertex<Integer>>() {
            public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
                if (o1.visited_p > o2.visited_p)
                    return -1;
                else if (o1.visited_p < o2.visited_p)
                    return 1;
                if (o1.getPerformance() > o2.getPerformance())
                    return -1;
                else if (o1.getPerformance() < o2.getPerformance())
                    return 1;
                return 0;
            }
        });

        return candidates;
    }
}







class EDGE1 {
    int cost, cap, v; // cost是边的费用，cap是边的容量，v是边终点的标号
    int next, re; // next是，re是反边终点的标号
    int current;
    boolean isPos;
}

class ZKW {
    final int MAXN = 2000;
    final int MAXM = 20000;
    final int INF = 10000000;

    static int DEMAND;

    EDGE1[] edge = new EDGE1[MAXM];
    int[] head = new int[MAXN]; // head表示的是从汇点出发到源点走过的路径，没走过的为-1，走过的为其他值
    int[] vis = new int[MAXN];
    int[] d = new int[MAXN]; // vis表示从源点出发到汇点走过的路径，走过的点标记为1，没走过的标记为0.
    int e;
    int ans, cost, src, des, n;
    int nt, m;
    int[][] dis = new int[MAXN][MAXN];
    char[][] s = new char[MAXN][MAXN];

    void init() {
        Arrays.fill(head, -1);
        e = 0;
        ans = cost = 0;
    }

    void add(int u, int v, int cap, int cost) {
        edge[e].current = u;
        edge[e].v = v;
        edge[e].cap = cap;
        edge[e].cost = cost;
        edge[e].re = e + 1;
        edge[e].next = head[u];
        edge[e].isPos = true;
        head[u] = e++;
        edge[e].current = v;
        edge[e].v = u;
        edge[e].cap = 0;
        edge[e].cost = -cost;
        edge[e].re = e - 1;
        edge[e].next = head[v];
        head[v] = e++;
        edge[e].isPos = false;

    }

    // f表示的是容量
    int aug(int u, int f) {
        if (u == des) {
            ans += cost * f;
            DEMAND += f;
            return f;
        }
        vis[u] = 1; // vis表示从源点出发，走过的点标记为1，没走过的标记为0.
        int tmp = f;
        for (int i = head[u]; i != -1; i = edge[i].next) {
            if (edge[i].cap != 0 && edge[i].cost == 0 && vis[edge[i].v] == 0) {
                int delta = aug(edge[i].v, tmp < edge[i].cap ? tmp : edge[i].cap);
                edge[i].cap -= delta;
                edge[edge[i].re].cap += delta;
                tmp -= delta;
                if (tmp == 0)
                    return f;
            }
        }
        return f - tmp;
    }

    int augNew(int u, int f) {
        if (u == des) {
            ans += cost * f;
            return f;
        }
        vis[u] = 1; // vis表示从源点出发，走过的点标记为1，没走过的标记为0.
        int tmp = f;
        for (int i = head[u]; i != -1; i = edge[i].next) {
            if (edge[i].cap != 0 && edge[i].cost == 0 && vis[edge[i].v] == 0) {
                int delta = augNew(edge[i].v, tmp < edge[i].cap ? tmp : edge[i].cap);
                edge[i].cap -= delta;
                edge[edge[i].re].cap += delta;
                tmp -= delta;
                if (tmp == 0)
                    return f;
            }
        }
        return f - tmp;
    }

    // 从汇点到源点的路径入队列的顺序
    boolean modlabel() {
        for (int i = 0; i <= n; i++)
            d[i] = INF; // d变量是Cij，即此刻的状态到下一刻状态的费用
        d[des] = 0;
        Deque<Integer> Q = new ArrayDeque<Integer>();
        Q.addLast(des);// (des);
        while (!Q.isEmpty()) {
            int u = Q.getFirst(), tmp;
            Q.pollFirst();

            for (int i = head[u]; i != -1; i = edge[i].next) {
                tmp = d[u] - edge[i].cost;
                if (((edge[edge[i].re].cap) != 0) && (tmp < d[edge[i].v]))// 判断是否满足ZKW算法的条件一
                {
                    d[edge[i].v] = tmp;
                    if ((d[edge[i].v]) <= d[Q.isEmpty() ? src : Q.getFirst()]) {
                        Q.addFirst(edge[i].v);
                    } else {
                        Q.addLast(edge[i].v);
                    }
                }
            }
        }
        // 对走过路径的费用清零
        for (int u = 1; u <= n; u++)
            for (int i = head[u]; i != -1; i = edge[i].next)
                edge[i].cost += d[edge[i].v] - d[u];
        cost += d[src];
        return d[src] < INF; // 如果为真，从汇点到源点找到一条可行路径
    }

    void costflow() {
        while (modlabel()) {
            do {
                Arrays.fill(vis, 0);
            } while (aug(src, INF) == 0);
        }
    }

    public String[] findWalk(int[][] weight,
                             boolean[] servers,
                             LinkedList<Vertex<Integer>> clients,
                             int count,
                             int num) {

        int[][] current_weight = new int[weight.length][weight.length];
        int[][] source_weight = new int[weight.length][weight.length];

        int[] path = new int[weight.length];
        boolean[] visited = new boolean[weight.length];
        boolean[] isClient = new boolean[weight.length];
//        LinkedList<Integer> stack = new LinkedList<Integer>();
        for (int i = 0; i < weight.length; i++) {
            source_weight[i] = weight[i].clone();
        }

        for (Vertex<Integer> client : clients){
            isClient[client.id] = true;
        }
        LinkedList<String> walks_set = new LinkedList<String>();

        while (modlabel()) {
            do {
                Arrays.fill(vis, 0);
            } while (augNew(src, INF) == 0);
        }

        for(int i = 0; i < count; i ++){
            current_weight[edge[i*2].current-1][edge[i*2].v-1] =
                    source_weight[edge[i*2].current-1][edge[i*2].v-1] - edge[i*2].cap;
        }
        int demand;
        int support;
        for(Vertex<Integer> client : clients){
            demand = client.getDemand();
            while (demand > 0){
                visited = new boolean[weight.length];
                path = new int[weight.length];
                dfs(client.id,
                        current_weight,
                        visited,
                        path,
                        demand,
                        0,
                        servers,
                        isClient,
                        walks_set,
                        num);
                String[] tmp = walks_set.getLast().split(" ");
                support = Integer.parseInt(tmp[tmp.length-1]);
                demand -= support;
            }
        }
        for(Vertex<Integer> client:clients){
            for(int i =0; i < weight.length; i++){
                if(current_weight[client.id][i]!=0)
                    System.out.println("error");
            }
        }
        String[] result = new String[walks_set.size()+2];
        result[0] = ""+walks_set.size();
        result[1] = "";
        for(int i = 0; i < walks_set.size(); i++){
            result[i+2] = walks_set.get(i);
        }
        return result;
    }

    public boolean dfs(int node, int[][] weights, boolean[] visited, int[] path, int demand, int support,
                       boolean[] servers,
                       boolean[] isClient,
                       LinkedList<String> walks_set,
                       int num){

        if(servers[node]){
            String tmp = node +" ";
            weights[path[node]][node] -= support;
            int u;
            for(u = path[node]; !isClient[u]; u = path[u]){
                tmp += u +" ";
                weights[path[u]][u] -= support;
            }
            tmp += (u-num+" "+support);
            walks_set.addLast(tmp);
//            System.out.println(tmp);
            return true;

        }else {
            visited[node] = true;
            for(int i = 0; i < weights.length; i ++){
                if(weights[node][i]!=0 && !visited[i]){
                    path[i] = node;
                    support = Math.min(demand,weights[node][i]);
                    if(dfs(i, weights, visited, path, support, support, servers, isClient, walks_set,num)){
                        visited[node] = false;
                        return true;
                    }
                }
            }
        }
        return true;
    }

    public static String[] getWalk(String[] graphContent, ArrayList<Integer> individual, int[][] weight,
                                   LinkedList<Vertex<Integer>> clients_init) {
        int count = 0;

        ZKW zkw = new ZKW();

        for (int i = 0; i < zkw.MAXM; i++) {
            EDGE1 e = new EDGE1();
            zkw.edge[i] = e;
        }

        zkw.init();

        // 添加边

        String[] nums = graphContent[0].split(" ");
        count = Integer.parseInt(nums[2]) + Integer.parseInt(nums[1]) * 2;
        int num = Integer.parseInt(nums[0]);
        zkw.n = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 2;
        zkw.src = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 1;
        zkw.des = zkw.n;

        for (int i = 4; i < Integer.parseInt(nums[1]) + 4; i++) {

            String[] data = graphContent[i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + 1, Integer.parseInt(data[1]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
            zkw.add(Integer.parseInt(data[1]) + 1, Integer.parseInt(data[0]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
        }
        for (int i = 0; i < Integer.parseInt(nums[2]); i++) {

            String[] data = graphContent[Integer.parseInt(nums[1]) + 5 + i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[1]) + 1,
                    Integer.parseInt(data[2]), 0);

        }
        for (int i = 0; i < Integer.parseInt(nums[2]); i++) {

            String[] data = graphContent[Integer.parseInt(nums[1]) + 5 + i].split(" ");

            zkw.add(zkw.src, Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[2]), 0);

        }

        boolean[] servers = new boolean[weight.length];
        for (Integer i : individual) {
            servers[i] = true;
            zkw.add(i + 1, zkw.des, zkw.INF, 0);
        }

        String[] content = zkw.findWalk(weight, servers, clients_init, count, num);

        return content;

    }

    public static int methodOfzkw1(String[] graphContent, ArrayList<Integer> servers) {
        ZKW zkw = new ZKW();
        // 初始化
        for (int i = 0; i < zkw.MAXM; i++) {
            EDGE1 e = new EDGE1();
            zkw.edge[i] = e;
        }

        Arrays.fill(zkw.head, -1);
        zkw.e = 0;
        zkw.ans = zkw.cost = 0;
        ZKW.DEMAND=0;//实际的供应需求
        // 添加边
        int demands = 0;//总需求
        String[] nums = graphContent[0].split(" ");

        //每个服务器的费用
        int costOfserver = Integer.parseInt(graphContent[2]);

        zkw.n = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 2;
        zkw.src = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 1;
        zkw.des = zkw.n;

        //添加边
        for (int i = 4; i < Integer.parseInt(nums[1]) + 4; i++) {
            String[] data = graphContent[i].split(" ");
            zkw.add(Integer.parseInt(data[0]) + 1, Integer.parseInt(data[1]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
            zkw.add(Integer.parseInt(data[1]) + 1, Integer.parseInt(data[0]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
        }
        for (int i = 0; i < Integer.parseInt(nums[2]); i++) {
            String[] data = graphContent[Integer.parseInt(nums[1]) + 5 + i].split(" ");
            zkw.add(Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[1]) + 1,
                    Integer.parseInt(data[2]), 0);
            demands += Integer.parseInt(data[2]);
            zkw.add(zkw.src, Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[2]), 0);
        }

        //添加超级汇点
        for (int i : servers) {
            zkw.add(i + 1, zkw.des, zkw.INF, 0);
        }

        zkw.costflow();

        //获取使用的服务器
        ArrayList<Integer> useOfServers = new ArrayList<Integer>();
        for (int i : servers) {
            if (zkw.edge[zkw.head[i + 1]].cap != zkw.INF) {
                useOfServers.add(i);
            }
        }
        int cost;
        if (demands != ZKW.DEMAND) {
            cost = -1;
        } else {
            cost = zkw.ans + costOfserver * (useOfServers.size());
        }

        // System.out.println("使用的服务器个数："+useOfServers.size());
        // System.out.println(useOfServers);
        //
        // System.out.println("最小费用："+cost);
        // System.out.println("供应需求："+zkw.demand);
        // System.out.println("总需求:"+demands);

        return cost;
    }

    //用于最后精简服务器的个数。如果减去一个服务器费用变少了就返回。
    public static ArrayList<Integer> methodOfzkw(String[] graphContent, ArrayList<Integer> servers) {
        int min = methodOfzkw1(graphContent, servers);
        int s = -1;
        for (int i = 0; i < servers.size(); i++) {
            int temp = servers.get(i);
            servers.remove(i);

            int t = methodOfzkw1(graphContent, servers);

            if (t == -1) {
                servers.add(i, temp);
                continue;
            }
            if (t < min) {
                min = t;
                s = i;
            }
            servers.add(i, temp);
        }
        if (s != -1) {
            servers.remove(s);
        }
//		System.out.println("最终费用："+min);
//		System.out.println(servers);
        return servers;

    }

}




































/*
 * 优化寻找服务器
 */
class Optimization2{

    /*
     * 种群基本信息配置 best_individual:最优个体 global_best_fitness：最有适应度值 gen_size：染色体大小
     * pop_size:种群大小 max_gen:最大迭代次数 mutation_rate:变异概率 candidates：候选种子集
     * population:种群
     */
    private LinkedList<Integer> certain_servers = new LinkedList<Integer>();
    private Chromosome sample_individual;
    static String[] graphContent;
    private Chromosome best_individual;
    private int global_best_fitness = Integer.MAX_VALUE;
    private int gen_size;
    private int pop_size;
    private final int max_gen = 100000;
    private int mate_size;
    private final double mutation_rate = 1.0;
    static private ArrayList<Vertex<Integer>> candidates;
    private ArrayList<Chromosome> population;
    private int end_condition;
    /*
     * performance_init：性价比 outputs_init:每个节点的输出带宽 weights_init：两个边上的带宽
     * clients_init:消费节点
     *
     */
    static private double max_performance = 0;
    static private double min_performance = Integer.MAX_VALUE;
    static private double[] normalization_performance;

    static private int total_visited_num = 0;
    static private int[] visited_num;
    static double[] visited_p;
    static private int[][] values_init;
    static private double[] performance_init;
    static private int[] outputs_init;
    static private int[][] weights_init;
    static private LinkedList<Vertex<Integer>> clients_init;
    static Graph G;

    // 计时器
    long start_time;
    long deadline;

    double FLAG_RATE;
    //
    public Optimization2(Graph G, long start_time, long deadline, String[] graphContent,double flag) {
        this.deadline = deadline;
        this.start_time = start_time;
        this.G = G;
        this.graphContent = graphContent;
        this.FLAG_RATE = flag;
    }

    // 测试
    public void test(ArrayList<Vertex<Integer>> candidates) {
        int min_cost = Integer.MAX_VALUE;
        int tmp_cost = 0;
        this.candidates = candidates;
        this.gen_size = G.getNumberOfNodes();
        visited_num = new int[G.getNumberOfNodes()];
        visited_p = new double[G.getNumberOfNodes()];
        best_individual = null;
        normalization_performance = new double[G.getNumberOfNodes()];
        performance_init = new double[G.getNumberOfVertexes()];
        outputs_init = new int[G.getNumberOfVertexes()];
        weights_init = new int[G.getNumberOfVertexes()][G.getNumberOfVertexes()];
        clients_init = G.getClients();

        // 对消费者排序
        Collections.sort(clients_init, new Comparator<Vertex<Integer>>() {
            public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
                if (o1.getDemand() > o2.getDemand())
                    return -1;
                else if (o1.getDemand() < o2.getDemand())
                    return 1;
                else
                    return 0;
            }
        });
        //
        for (int i = 0; i < G.getNumberOfVertexes(); i++) {
            Vertex<Integer> node = G.getNode(i);
            performance_init[i] = node.getPerformance();
            outputs_init[i] = node.getOutputs();
            LinkedList<Vertex<Integer>> neighbors = node.getNeighbors();
            for (Vertex<Integer> neighbor : neighbors) {
                if (neighbor.isNode())
                    weights_init[i][neighbor.id] = node.getWeight(neighbor);
                if (node.isClient())
                    weights_init[node.id][neighbor.id] = node.getDemand();
            }
            if (node.isNode()) {
                node.sortNeighbors();
                if (performance_init[i] > max_performance) {
                    max_performance = performance_init[i];
                } else if (performance_init[i] < min_performance) {
                    min_performance = performance_init[i];
                }
            }
        }
        // 归一化
        for (int i = 0; i < normalization_performance.length; i++) {
            normalization_performance[i] = (performance_init[i] - min_performance)
                    / (max_performance - min_performance);
        }

        //
        sample_individual = new Chromosome();
        sample_individual.initGeneSize(gen_size);

        int idx;
        LinkedList<Vertex<Integer>> set = new LinkedList<Vertex<Integer>>();
        for (Vertex<Integer> client : clients_init) {

            idx = client.getNeighbors().getFirst().id;

            sample_individual.gene[idx] = true;
            set.add(client.getNeighbors().getFirst());

        }
        // 此处基于performance与priotiry相同
        Collections.sort(set, new Comparator<Vertex<Integer>>() {
            public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
                if (o1.getPerformance() > o2.getPerformance())
                    return 1;
                else if (o1.getPerformance() < o2.getPerformance())
                    return -1;
                return 0;
            }
        });
        for (int i = 0; i < set.size(); i++) {
            sample_individual.gene[set.get(i).id] = false;
            tmp_cost = getCostBk(sample_individual);
            if (tmp_cost < min_cost) {
                min_cost = tmp_cost;

            } else {
                sample_individual.gene[set.get(i).id] = true;
            }
        }

        for(int i=0; i<sample_individual.gene.length;i++){
            if(sample_individual.gene[i]){
                sample_individual.gene[i] = false;
                tmp_cost = getCostBk(sample_individual);
                if(tmp_cost < min_cost){
                    min_cost = tmp_cost;
                }else {
                    sample_individual.gene[i] = true;
                    certain_servers.add(i);
                }
            }
        }

    }

    public void setup(ArrayList<Vertex<Integer>> candidates) {
        int min_cost = Integer.MAX_VALUE;
        int tmp_cost = 0;
        this.candidates = candidates;
        this.gen_size = G.getNumberOfNodes();
        visited_num = new int[G.getNumberOfNodes()];
        visited_p = new double[G.getNumberOfNodes()];
        best_individual = null;
        normalization_performance = new double[G.getNumberOfNodes()];
        performance_init = new double[G.getNumberOfVertexes()];
        outputs_init = new int[G.getNumberOfVertexes()];
        weights_init = new int[G.getNumberOfVertexes()][G.getNumberOfVertexes()];
        clients_init = G.getClients();

        // 对消费者排序
        Collections.sort(clients_init, new Comparator<Vertex<Integer>>() {
            public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
                if (o1.getDemand() > o2.getDemand())
                    return -1;
                else if (o1.getDemand() < o2.getDemand())
                    return 1;
                else
                    return 0;
            }
        });
        //
        for (int i = 0; i < G.getNumberOfVertexes(); i++) {
            Vertex<Integer> node = G.getNode(i);
            performance_init[i] = node.getPerformance();
            outputs_init[i] = node.getOutputs();
            LinkedList<Vertex<Integer>> neighbors = node.getNeighbors();
            for (Vertex<Integer> neighbor : neighbors) {
                if (neighbor.isNode())
                    weights_init[i][neighbor.id] = node.getWeight(neighbor);
                if (node.isClient())
                    weights_init[node.id][neighbor.id] = node.getDemand();
            }
            if (node.isNode()) {
                node.sortNeighbors();
                if (performance_init[i] > max_performance) {
                    max_performance = performance_init[i];
                } else if (performance_init[i] < min_performance) {
                    min_performance = performance_init[i];
                }
            }
        }
        // 归一化
        for (int i = 0; i < normalization_performance.length; i++) {
            normalization_performance[i] = (performance_init[i] - min_performance)
                    / (max_performance - min_performance);
        }

        //
        sample_individual = new Chromosome();
        sample_individual.initGeneSize(gen_size);

        int idx;
        LinkedList<Vertex<Integer>> set = new LinkedList<Vertex<Integer>>();
//        for (Vertex<Integer> client : clients_init) {
        for (int i=0;i< clients_init.size()*FLAG_RATE;i++) {
//            idx = client.getNeighbors().getFirst().id;
//
//            sample_individual.gene[idx] = true;
//            set.add(client.getNeighbors().getFirst());
            idx = clients_init.get(i).getNeighbors().getFirst().id;
            sample_individual.gene[idx] = true;
            set.add(clients_init.get(i).getNeighbors().getFirst());
            certain_servers.add(idx);

        }
        // 此处基于performance与priotiry相同
        Collections.sort(set, new Comparator<Vertex<Integer>>() {
            public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
                if (o1.getPerformance() > o2.getPerformance())
                    return 1;
                else if (o1.getPerformance() < o2.getPerformance())
                    return -1;
                return 0;
            }
        });
        for (int i = 0; i < set.size(); i++) {
            sample_individual.gene[set.get(i).id] = false;
            tmp_cost = getCostBk(sample_individual);
            if (tmp_cost < min_cost) {
                min_cost = tmp_cost;
                //certain_servers.add(set.get(i).id);
            } else {
                sample_individual.gene[set.get(i).id] = true;
            }
        }
        for(int i=0; i<sample_individual.gene.length;i++){
            if(sample_individual.gene[i]){
                sample_individual.gene[i] = false;
                tmp_cost = getCostBk(sample_individual);
                if(tmp_cost < min_cost){
                    min_cost = tmp_cost;
                }else {
                    sample_individual.gene[i] = true;
                    //certain_servers.add(i);
                }
            }
        }
        verify(sample_individual);
    }

    //判断是否可行解
    public void verify(Chromosome individual){
        ArrayList<Integer> servers =new ArrayList<Integer>();
        for(int i = 0; i < individual.gene.length; i++){
            if(individual.gene[i])
                servers.add(i);
        }
        Result base = ZKW2.methodOfzkw2(graphContent,servers);
        if(base.support!=base.demand){
            modify(individual);
        }
    }

    //修正，不可行解
    public int modify(Chromosome individual) {
        int result = 0;
        boolean[] gen = individual.getGene();
        // server_set：服务器集合
        HashSet<Vertex<Integer>> server_set = new HashSet<Vertex<Integer>>();
        HashSet<Integer> sets = new HashSet<Integer>();
        for (int idx = 0; idx < gen.length; idx++) {
            if (gen[idx])
                server_set.add(G.getNode(idx));
        }
        // 寻找该解下的所有路径
        LinkedList<LinkedList<Step>> walks = findWalks(G, server_set, clients_init, performance_init, outputs_init,
                weights_init, G.getNumberOfVertexes());
        individual.gene = new boolean[individual.size];
        // 计算成本

        for (LinkedList<Step> walk : walks) {
            for (Step step : walk) {
                if (server_set.contains(step.node)) {
                    sets.add(step.node.id);
                    result += step.cost;
                    break;
                }
            }
        }
        for (Integer i : sets) {
            individual.gene[i] = true;
        }
        result += G.getServerCost() * sets.size();

        return result;
    }


    // 种群初始化
    public void init(int par) {
        long end_time;
        int min_cost = Integer.MAX_VALUE;
        int tmp_cost = 0;
        best_individual = null;

        // 初始化种群
        population = new ArrayList<Chromosome>();
        int idx = 0;
        min_cost = Integer.MAX_VALUE;
        tmp_cost = 0;
        if (par == 1) {
            pop_size = 4;
            mate_size = (int) (pop_size * 0.5);
            end_condition = 10000000;
        } else if (par == 0) {
            pop_size = 50;
            mate_size = (int) (pop_size * 0.5);
            end_condition = 10000000;
        } else {
            pop_size = 100;
            mate_size = (int) (pop_size * 0.5);
            end_condition = 200;
        }
        for (int i = 0; i < pop_size; i++) {
            Chromosome individual = null;
            if (i < (int) (pop_size * 0.8)) {
                individual = new Chromosome();
                individual.initGeneSize(gen_size);
                for (int j = 0; j < sample_individual.gene.length; j++) {
                    individual.gene[j] = sample_individual.gene[j];
                }
            }
            else{
                individual = new Chromosome(gen_size,
                        normalization_performance,
                        clients_init,
                        candidates,
                        (float) 0.6,
                        sample_individual, certain_servers);
            }

            tmp_cost = getCostBk(individual);
            individual.fitness_score = tmp_cost;
            population.add(individual);
            if (min_cost > tmp_cost) {
                min_cost = tmp_cost;
                best_individual = Chromosome.clone(individual);
                best_individual.fitness_score = min_cost;
                global_best_fitness = min_cost;
            }
        }
    }

    // 选择
    public ArrayList<Chromosome> selection() {
        Collections.sort(population, new Comparator<Chromosome>() {
            public int compare(Chromosome o1, Chromosome o2) {
                if(o1.fitness_score > o2.fitness_score)
                    return 1;
                else if(o1.fitness_score < o2.fitness_score)
                    return -1;
                return 0;
            }
        });
        ArrayList<Chromosome> next_population = new ArrayList<Chromosome>();
        next_population.add(best_individual);
        for(int i =0; i <pop_size*0.0; i++){
            next_population.add(Chromosome.clone(population.get(i)));
        }
        while (next_population.size() < mate_size) {
            Random rand = new Random();
            int i = rand.nextInt(pop_size);
            int j = rand.nextInt(pop_size);
            if (population.get(i).fitness_score < population.get(j).fitness_score)
                next_population.add(Chromosome.clone(population.get(i)));
            else
                next_population.add(Chromosome.clone(population.get(j)));
        }
        return next_population;
    }

    // 交叉
    public void crossover(ArrayList<Chromosome> next_population) {
        Chromosome a;
        Chromosome b;
        int index;
        Chromosome a_next;
        Chromosome b_next;
        while (next_population.size() < pop_size) {
            Random random = new Random();
            a = next_population.get(random.nextInt(mate_size));
            b = next_population.get(random.nextInt(mate_size));
            index = random.nextInt(gen_size);
            a_next = new Chromosome();
            b_next = new Chromosome();
            a_next.initGeneSize(gen_size);
            b_next.initGeneSize(gen_size);
            for (int i = 0; i < index; i++) {
                a_next.gene[i] = a.gene[i];
                b_next.gene[i] = b.gene[i];
            }
            for (int i = index; i < gen_size; i++) {
                a_next.gene[i] = b.gene[i];
                b_next.gene[i] = a.gene[i];
            }
            next_population.add(a_next);
            next_population.add(b_next);
        }
    }

    /*
    // 变异
    public void mutate(ArrayList<Chromosome> next_population) {
        Chromosome individual;
        Random random = new Random();
        Vertex<Integer> node;
        int idx;
        for (int i = 1; i < next_population.size(); i++) {
            individual = next_population.get(i);
            for(int t =0; t < 2; t++){
                individual.gene[certain_servers.get(random.nextInt(certain_servers.size()))] =
                        !individual.gene[certain_servers.get(random.nextInt(certain_servers.size()))];
            }
        }
    }
    */

    // 变异

    public void mutate(ArrayList<Chromosome> next_population) {
        Chromosome individual;
        Random random = new Random();
        Vertex<Integer> node;
        int idx;
        while(next_population.size()!=pop_size){
            individual = Chromosome.clone(next_population.get(random.nextInt(mate_size)));
            for(int i =0; i < 2; i++){
                individual.gene[certain_servers.get(random.nextInt(certain_servers.size()))] =
                        Math.random()>0.5;
            }
            next_population.add(individual);
        }
    }

    // 局部搜索
    public void localSearch() {
        Chromosome tmp = Chromosome.clone(best_individual);
        Random random = new Random();
        int min_cost = global_best_fitness;
        int tmp_cost = 0;

        int idx1 = certain_servers.get(random.nextInt(certain_servers.size()));
        int idx2 = certain_servers.get(random.nextInt(certain_servers.size()));
        //int idx3 = certain_servers.get(random.nextInt(certain_servers.size()));
        //int idx4 = certain_servers.get(random.nextInt(certain_servers.size()));
        //int idx5 = certain_servers.get(random.nextInt(certain_servers.size()));

        tmp.gene[idx1] = Math.random()>0.5;
        tmp.gene[idx2] = Math.random()>0.5;
        //tmp.gene[idx3] = Math.random()>0.5;
        //tmp.gene[idx4] = Math.random()>0.5;
        //tmp.gene[idx5] = Math.random()>0.5;
        tmp_cost = getCost(tmp);
        if(tmp_cost < min_cost){
//            System.out.println("+++++");
            min_cost = getCost(tmp);
            best_individual = Chromosome.clone(tmp);
            best_individual.fitness_score = min_cost;
            global_best_fitness = min_cost;
        }else {
            tmp.gene[idx1] = !tmp.gene[idx1];
            tmp.gene[idx2] = !tmp.gene[idx2];
            //tmp.gene[idx3] = !tmp.gene[idx3];
            //tmp.gene[idx4] = !tmp.gene[idx4];
            //tmp.gene[idx5] = !tmp.gene[idx5];
        }


    }
    // 获得下一代
    public ArrayList<Chromosome> genNextPopulation() {
        ArrayList<Chromosome> next_population = selection();
        //crossover(next_population);
        mutate(next_population);
        localSearch();
        return next_population;
    }

    // 训练
    public Chromosome train() {
        LinkedList<Integer> test;
        int count = 0;
        int last_min_cost = 0;
        int min_cost = Integer.MAX_VALUE;
        int tmp_cost;
        for (int t = 0; t < max_gen; t++) {
            tmp_cost = 0;
            if(count>1){
                localSearch();
            }else {
                population = genNextPopulation();
                for (Chromosome individual : population) {
                    tmp_cost = getCost(individual);
                    individual.fitness_score = tmp_cost;
                    if (min_cost > tmp_cost) {
                        min_cost = tmp_cost;
                        best_individual = Chromosome.clone(individual);
                        best_individual.fitness_score = min_cost;
                        global_best_fitness = min_cost;
                    }
                }
            }
            // 判断是否收敛
            if (last_min_cost == global_best_fitness) {
                count += 1;
                if (count == end_condition) {
                    return best_individual;
                }
            } else {
                count = 0;
            }
            last_min_cost = min_cost;
            Long end_time = System.currentTimeMillis();
            // 判断是否超时
            if (end_time - start_time > deadline) {
                return best_individual;
            }
//            System.out.println(t + "   " + global_best_fitness);
        }

        return best_individual;
    }

    // 获得结果
    public static String[] getResult(Chromosome individual) {
        int idx = 0;

        String tmp;
        boolean[] gen = individual.getGene();
        ArrayList<Integer> server_set = new ArrayList<Integer>();
        for (int i = 0; i < gen.length; i++) {
            if (gen[i])
                server_set.add(G.getNode(i).id);///// 获取最优服务器位置
        }
//        String[] result = ZKW2.getWalk(graphContent, ZKW2.methodOfzkw(graphContent, server_set), weights_init,
        //clients_init);
        String[] result = ZKW2.getWalk(graphContent, ZKW2.methodOfzkw(graphContent, server_set), weights_init, clients_init);
        return result;
    }

    // 目标函数，获得每个解得成本
    public int getCost(Chromosome individual) {
        int result = 0;
        boolean[] gen = individual.getGene();
        // server_set：服务器集合
        ArrayList<Integer> servers =new ArrayList<Integer>();
        for (int idx = 0; idx < gen.length; idx++) {
            if (gen[idx])
                servers.add(idx);
        }
        result= ZKW2.methodOfzkw1(graphContent, servers);
        return result;
    }

    public int getCostBk(Chromosome individual) {
        int result = 0;
        boolean[] gen = individual.getGene();
        // server_set：服务器集合
        HashSet<Vertex<Integer>> server_set = new HashSet<Vertex<Integer>>();
        HashSet<Integer> sets = new HashSet<Integer>();
        for (int idx = 0; idx < gen.length; idx++) {
            if (gen[idx])
                server_set.add(G.getNode(idx));
        }
        // 寻找该解下的所有路径
        LinkedList<LinkedList<Step>> walks = findWalks(G, server_set, clients_init, performance_init, outputs_init,
                weights_init, G.getNumberOfVertexes());
        ArrayList<Integer> servers=new ArrayList<Integer>();
        for(Vertex<Integer> i:server_set){
            servers.add(i.id);
        }
        for (LinkedList<Step> walk : walks) {
            for (Step step : walk) {

                if (server_set.contains(step.node)) {
                    sets.add(step.node.id);
                    result += step.cost;
                    break;
                }
            }
        }
        result += G.getServerCost() * sets.size();
        return result;
    }

    // 寻找所有路径
    public static LinkedList<LinkedList<Step>> findWalks(Graph G, HashSet<Vertex<Integer>> server_set,
                                                         LinkedList<Vertex<Integer>> clients, double[] performance, int[] outputs, int[][] weights, int len) {

        LinkedList<LinkedList<Step>> walks = new LinkedList<LinkedList<Step>>();
        // outputs_neighbor_server，该节点的邻居节点有服务器，计算其与服务器的带宽和
        // neighbor_has_server：判断该节点的邻居节点是否有服务器
        boolean[] neighbor_has_server = new boolean[len];
        int[] outputs_neighbor_server = new int[len];
        // 克隆信息
        double[] performance_tmp = performance;
        int[] outputs_tmp = outputs.clone();
        int[][] weights_tmp = new int[len][];
        for (int i = 0; i < len; i++) {
            weights_tmp[i] = weights[i].clone();
        }
        // 根据服务器，更新网络中各个节点的性价比
        for (Vertex<Integer> server : server_set) {
            for (Vertex<Integer> neighbor : server.getNeighbors()) {
                if (neighbor.isNode() && !server_set.contains(neighbor)) {
                    neighbor_has_server[neighbor.id] = true;
                    outputs_neighbor_server[neighbor.id] += neighbor.getWeight(server);
                }
            }
        }
        // 为每一个服务器，寻找路径
        for (Vertex<Integer> client : clients) {
            int demand = client.getDemand();
            int remain_demand = demand;
            // 确保满足每一个服务器的需求带宽
            while (remain_demand != 0) {
                LinkedList<Step> walk = find(client, client.getDemand(), server_set, neighbor_has_server,
                        performance_tmp, weights_tmp, outputs_neighbor_server, outputs_tmp);

                walks.addLast(walk);
                remain_demand -= walk.getLast().demand;
            }
        }

        return walks;
    }

    // 给定服务器和需要承担的带宽，寻找路径
    public static LinkedList<Step> find(Vertex<Integer> client, int demand, final HashSet<Vertex<Integer>> server_set,
                                        final boolean[] neighbor_has_server, final double[] performance, int[][] weights,
                                        int[] outputs_neighbor_server, int[] outputs) {

        HashSet<Vertex<Integer>> selected_nodes = new HashSet<Vertex<Integer>>();
        int cost = 0;
        LinkedList<Step> walk = new LinkedList<Step>();
        walk.addLast(new Step(client, demand, 0));

        // 用于加快排序的参数
        int idx_0 = 0;
        int idx_1 = 0;
        boolean flag_sort = true;
        Vertex<Integer> tmp_node;
        // 用于控制路径寻找深度
        int count = 0;
        final int max_find_count = 6;

        boolean flag = true;
        while (flag) {
            Step current_step = walk.getLast();
            final Vertex<Integer> current_node = current_step.node;
            int current_demand = current_step.demand;
            // 生成新的邻居节点集
            LinkedList<Vertex<Integer>> raw_neighbors = current_node.getNeighbors();
            LinkedList<Vertex<Integer>> neighbors = new LinkedList<Vertex<Integer>>();

            for (Vertex<Integer> neighbor : raw_neighbors) {
                if (neighbor.isNode() && weights[current_node.id][neighbor.id] > 0
                        && !selected_nodes.contains(neighbor)) {

                    if (server_set.contains(neighbor)) {
                        for (int i = 0; i < neighbors.size(); i++) {
                            tmp_node = neighbors.get(i);
                            if (server_set.contains(tmp_node)) {
                                if (neighbor.getValue(current_node) < tmp_node.getValue(current_node)) {
                                    neighbors.add(i, neighbor);
                                    flag_sort = false;
                                    break;
                                }
                            } else {
                                neighbors.add(i, neighbor);
                                flag_sort = false;
                                break;
                            }
                        }
                        if (flag_sort)
                            neighbors.addLast(neighbor);
                        flag_sort = true;
                        idx_0 += 1;
                        idx_1 += 1;
                    } else if (neighbor_has_server[neighbor.id]) {
                        for (int i = idx_0; i < neighbors.size(); i++) {
                            tmp_node = neighbors.get(i);
                            if (server_set.contains(tmp_node)) {
                                ;
                            } else if (neighbor_has_server[tmp_node.id]) {
                                if (neighbor.getValue(current_node) < tmp_node.getValue(current_node)) {
                                    neighbors.add(i, neighbor);
                                    flag_sort = false;
                                    break;
                                }
                            } else {
                                neighbors.add(i, neighbor);
                                flag_sort = false;
                                break;
                            }
                        }
                        if (flag_sort)
                            neighbors.addLast(neighbor);
                        idx_1 += 1;
                        flag_sort = true;
                    } else {
                        for (int i = idx_1; i < neighbors.size(); i++) {
                            tmp_node = neighbors.get(i);
                            if (server_set.contains(tmp_node)) {
                                ;
                            } else if (neighbor_has_server[tmp_node.id]) {
                                ;
                            } else {
                                if (neighbor.getValue(current_node) < tmp_node.getValue(current_node)) {
                                    neighbors.add(i, neighbor);
                                    flag_sort = false;
                                    break;
                                }
                            }
                        }
                        if (flag_sort)
                            neighbors.addLast(neighbor);
                        flag_sort = true;
                    }
                }
            }
			/*
			 * System.out.println("开始"); System.out.println(server_set);
			 * System.out.println(selected_nodes);
			 * System.out.println(raw_neighbors);
			 * System.out.println(current_node); System.out.println("测试一");
			 * for(Vertex<Integer> neighbor : neighbors){
			 * System.out.println(neighbor+" "+neighbor.getPerformance()+"  "
			 * +neighbor.getValue(current_node)); }
			 */
            // 根据规则，排序
			/*
			 * neighbors = new LinkedList<Vertex<Integer>>(); for
			 * (Vertex<Integer> neighbor : raw_neighbors) { if
			 * (neighbor.isNode() && weights[current_node.id][neighbor.id] > 0
			 * && !selected_nodes.contains(neighbor)){
			 * neighbors.addLast(neighbor); } } Collections.sort(neighbors, new
			 * Comparator<Vertex<Integer>>() { public int
			 * compare(Vertex<Integer> o1, Vertex<Integer> o2) {
			 * if(server_set.contains(o1) && !server_set.contains(o2)) return
			 * -1; else if((!server_set.contains(o1) &&
			 * server_set.contains(o2))) return 1;
			 *
			 * if ((o1.isNode() && neighbor_has_server[o1.id] && o2.isNode() &&
			 * !neighbor_has_server[o2.id]) ) return -1; else if((o1.isNode() &&
			 * !neighbor_has_server[o1.id] && o2.isNode() &&
			 * neighbor_has_server[o2.id])) return 1;
			 *
			 * if(o1.getValue(current_node) > o2.getValue(current_node)) return
			 * 1; if(o1.getValue(current_node) < o2.getValue(current_node))
			 * return -1;
			 *
			 * if(performance[o1.id] > performance[o2.id]) return -1; else
			 * if(performance[o1.id] < performance[o2.id]) return 1; return 0; }
			 * });
			 */
			/*
			 * //System.out.println("测试二"); //for(Vertex<Integer> neighbor :
			 * neighbors){ //
			 * System.out.println(neighbor+" "+neighbor.getPerformance()+"  "
			 * +neighbor.getValue(current_node)); //}
			 */
            // 寻找下一步
            Step next_step = findNextStep(current_node, current_demand, neighbors, server_set, neighbor_has_server,
                    weights, outputs_neighbor_server, performance, outputs);
            // 判断你是否停止寻找节点，终止搜索。并且计算每一步的成本。
            // 在终止前，应该更新网络属性。例如，outputs，weights和perforamnce
            if (next_step != null) {
                walk.addLast(next_step);
                selected_nodes.add(next_step.node);
                count += 1;
                if (server_set.contains(next_step.node)) {
                    flag = false;
                    int size = walk.size();
                    Vertex<Integer> last = client;
                    for (int i = 1; i < size; i++) {
                        Step step = walk.get(i);
                        if (i > 1) {
                            cost += walk.getLast().demand * step.value;
                            step.cost = cost;
                        }
                        weights[last.id][step.node.id] -= walk.getLast().demand;
                        if (neighbor_has_server[step.node.id])
                            outputs_neighbor_server[step.node.id] -= walk.getLast().demand;
                        outputs[step.node.id] -= walk.getLast().demand;
                        last = step.node;
                    }
                } else if (count > max_find_count) {
                    flag = false;
                    int size = walk.size();
                    Vertex<Integer> last = client;
                    for (int i = 1; i < size; i++) {
                        Step step = walk.get(i);
                        if (i > 1) {
                            cost += walk.getLast().demand * step.value;
                            step.cost = cost;
                        }
                        weights[last.id][step.node.id] -= walk.getLast().demand;
                        if (neighbor_has_server[step.node.id])
                            outputs_neighbor_server[step.node.id] -= walk.getLast().demand;
                        outputs[step.node.id] -= walk.getLast().demand;
                        last = step.node;
                    }
                    // 更新，新增服务器
                    Vertex<Integer> new_server = walk.getLast().node;
                    server_set.add(new_server);
                    // 更新新增服务器的邻居节点的配置信息，如perforamnce，outputs_neighbor_server
                    for (Vertex<Integer> neighbor : new_server.getNeighbors()) {
                        if (neighbor.isNode() && !server_set.contains(neighbor)) {
                            neighbor_has_server[neighbor.id] = true;
                            outputs_neighbor_server[neighbor.id] += weights[neighbor.id][new_server.id];
                        }
                    }
                }
            } else {
                flag = false;
                int size = walk.size();
                Vertex<Integer> last = client;
                for (int i = 1; i < size; i++) {
                    Step step = walk.get(i);
                    if (i > 1) {
                        cost += walk.getLast().demand * step.value;
                        step.cost = cost;
                    }
                    weights[last.id][step.node.id] -= walk.getLast().demand;
                    if (neighbor_has_server[step.node.id])
                        outputs_neighbor_server[step.node.id] -= walk.getLast().demand;
                    outputs[step.node.id] -= walk.getLast().demand;
                    last = step.node;
                }
                // 更新，新增服务器
                Vertex<Integer> new_server = walk.getLast().node;
                server_set.add(new_server);
                // 更新新增服务器的邻居节点的配置信息，如perforamnce，outputs_neighbor_server
                for (Vertex<Integer> neighbor : new_server.getNeighbors()) {
                    if (neighbor.isNode() && !server_set.contains(neighbor)) {
                        neighbor_has_server[neighbor.id] = true;
                        outputs_neighbor_server[neighbor.id] += weights[neighbor.id][new_server.id];
                    }
                }
            }
        }
        return walk;
    }

    // 寻找一条路径中的下一步骤
    public static Step findNextStep(Vertex<Integer> current_node, int current_demand,
                                    LinkedList<Vertex<Integer>> neighbors, final HashSet<Vertex<Integer>> server_set,
                                    boolean[] neighbor_has_server, int[][] weights, int[] outputs_neighbor_server, double[] performance,
                                    int[] outputs) {

        if (neighbors.isEmpty())
            return null;
        Vertex<Integer> neighbor = neighbors.removeFirst();

        if (server_set.contains(neighbor) && weights[current_node.id][neighbor.id] > 0) {
            int support = Math.min(current_demand, weights[current_node.id][neighbor.id]);
            return new Step(neighbor, support, current_node.getValue(neighbor));
        } else if (neighbor_has_server[neighbor.id] && outputs_neighbor_server[neighbor.id] > 0) {
            int server_support = Math.min(outputs_neighbor_server[neighbor.id], weights[current_node.id][neighbor.id]);
            int support = Math.min(current_demand, server_support);

            return new Step(neighbor, support, current_node.getValue(neighbor));
        } else if (performance[neighbor.id] > performance[current_node.id]
                && weights[current_node.id][neighbor.id] > 0) {
            int node_support = Math.min(outputs[neighbor.id], weights[current_node.id][neighbor.id]);
            int support = Math.min(current_demand, node_support);
            return new Step(neighbor, support, current_node.getValue(neighbor));
        }
        return findNextStep(current_node, current_demand, neighbors, server_set, neighbor_has_server, weights,
                outputs_neighbor_server, performance, outputs);
    }

    // 影响力最大化，选择候选集

    public ArrayList<Vertex<Integer>> influenceMax(int keyNum) {

        ArrayList<Vertex<Integer>> maxVertexs = new ArrayList<Vertex<Integer>>();

        TreeSet<Vertex> sortVertexsByPriority = new TreeSet<Vertex>();
        // 此处是原方法的基于度的节点的存储方式
        ArrayList<Integer> vertexsOfNotClient = new ArrayList<Integer>();

        Vertex<Integer>[] vertexs = this.G.getVertexs();
        LinkedList<Vertex<Integer>> clients = this.G.getClients();

        for (Vertex v : vertexs) {
            if (clients.contains(v)) {
                continue;
            }
            sortVertexsByPriority.add(v);
            vertexsOfNotClient.add(v.id);
        }

        for (int j = 0; j < keyNum; j++) {
            Vertex v = sortVertexsByPriority.pollFirst();
            maxVertexs.add(v);
            LinkedList<Vertex> neighbor = v.getNeighbors();
            int value = 0;
            for (Vertex vn : neighbor) {
                if (!maxVertexs.contains(vn) && !clients.contains(vn)) {
                    sortVertexsByPriority.remove(vn);
                    vn.setPriority(vn.getPriority(), v.id);
                    sortVertexsByPriority.add(vn);
                }
            }

        }

        return maxVertexs;
    }

    public ArrayList<Vertex<Integer>> influenceMax() {
        ArrayList<Vertex<Integer>> candidates = new ArrayList<Vertex<Integer>>();
        int[][] value_init = new int[G.getNumberOfVertexes()][G.getNumberOfVertexes()];
        int[] visited_num_tmp = new int[G.getNumberOfVertexes()];
        int[] values = new int[G.getNumberOfVertexes()];
        LinkedList<Vertex<Integer>> neighbors;
        LinkedList<Vertex<Integer>> current_neighbors;
        LinkedList<Vertex<Integer>> next_neighbors;

        Vertex<Integer> node;
        for (int i = 0; i < G.getNumberOfNodes(); i++) {
            node = G.getNode(i);
            neighbors = node.getNeighbors();
            for (Vertex<Integer> neighbor : neighbors) {
                value_init[node.id][neighbor.id] = (int) node.getValue(neighbor);
            }
        }
        //
        for (Vertex<Integer> client : G.getClients()) {
            for (int idx = 0; idx < values.length; idx++) {
                values[idx] = Integer.MAX_VALUE;
            }
            current_neighbors = new LinkedList<Vertex<Integer>>();
            values[client.id] = 0;
            current_neighbors.add(client);
            for (int i = 0; i < 3; i++) {
                next_neighbors = new LinkedList<Vertex<Integer>>();
                while (!current_neighbors.isEmpty()) {
                    node = current_neighbors.removeFirst();
                    neighbors = node.getNeighbors();
                    for (Vertex<Integer> neighbor : neighbors) {

                        visited_num_tmp[neighbor.id] += 1;
                        next_neighbors.add(neighbor);
                    }
                }
                current_neighbors = next_neighbors;
            }
        }
        int max = 0;
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < G.getNumberOfNodes(); i++) {
            if (visited_num_tmp[i] != 0) {
                if (visited_num_tmp[i] > max) {
                    max = visited_num_tmp[i];
                }
                if (visited_num_tmp[i] < min) {
                    min = visited_num_tmp[i];
                }
            }
        }
        for (int i = 0; i < G.getNumberOfNodes(); i++) {
            G.getNode(i).visited_p = ((visited_num_tmp[i] - min) / (float) (max - min));
            candidates.add(G.getNode(i));

        }
        Collections.sort(candidates, new Comparator<Vertex<Integer>>() {
            public int compare(Vertex<Integer> o1, Vertex<Integer> o2) {
                if (o1.visited_p > o2.visited_p)
                    return -1;
                else if (o1.visited_p < o2.visited_p)
                    return 1;
                if (o1.getPerformance() > o2.getPerformance())
                    return -1;
                else if (o1.getPerformance() < o2.getPerformance())
                    return 1;
                return 0;
            }
        });

        return candidates;
    }
}


class EDGE2 {
    int cost, cap, v; // cost是边的费用，cap是边的容量，v是边终点的标号
    int next, re; // next是，re是反边终点的标号
    int current;
    boolean isPos;
}

class ZKW2 {

    final int MAXN = 2000;
    final int MAXM = 20000;
    final int INF = 10000000;

    static int DEMAND;

    EDGE2[] edge = new EDGE2[MAXM];
    int[] head = new int[MAXN]; // head表示的是从汇点出发到源点走过的路径，没走过的为-1，走过的为其他值
    int[] vis = new int[MAXN];
    int[] d = new int[MAXN]; // vis表示从源点出发到汇点走过的路径，走过的点标记为1，没走过的标记为0.
    int e;
    int ans, cost, src, des, n;
    int nt, m;
    int[][] dis = new int[MAXN][MAXN];
    char[][] s = new char[MAXN][MAXN];

    void init() {
        Arrays.fill(head, -1);
        e = 0;
        this.DEMAND = 0;
        ans = cost = 0;
    }

    void add(int u, int v, int cap, int cost) {
        edge[e].current = u;
        edge[e].v = v;
        edge[e].cap = cap;
        edge[e].cost = cost;
        edge[e].re = e + 1;
        edge[e].next = head[u];
        edge[e].isPos = true;
        head[u] = e++;
        edge[e].current = v;
        edge[e].v = u;
        edge[e].cap = 0;
        edge[e].cost = -cost;
        edge[e].re = e - 1;
        edge[e].next = head[v];
        head[v] = e++;
        edge[e].isPos = false;

    }

    // f表示的是容量
    int aug(int u, int f) {
        if (u == des) {
            ans += cost * f;
            DEMAND += f;
            return f;
        }
        vis[u] = 1; // vis表示从源点出发，走过的点标记为1，没走过的标记为0.
        int tmp = f;
        for (int i = head[u]; i != -1; i = edge[i].next) {
            if (edge[i].cap != 0 && edge[i].cost == 0 && vis[edge[i].v] == 0) {
                int delta = aug(edge[i].v, tmp < edge[i].cap ? tmp : edge[i].cap);
                edge[i].cap -= delta;
                edge[edge[i].re].cap += delta;
                tmp -= delta;
                if (tmp == 0)
                    return f;
            }
        }
        return f - tmp;
    }

    int augNew(int u, int f) {
        if (u == des) {
            ans += cost * f;
            DEMAND += f;
            return f;
        }
        vis[u] = 1; // vis表示从源点出发，走过的点标记为1，没走过的标记为0.
        int tmp = f;
        for (int i = head[u]; i != -1; i = edge[i].next) {
            if (edge[i].cap != 0 && edge[i].cost == 0 && vis[edge[i].v] == 0) {
                int delta = aug(edge[i].v, tmp < edge[i].cap ? tmp : edge[i].cap);
                edge[i].cap -= delta;
                edge[edge[i].re].cap += delta;
                tmp -= delta;
                if (tmp == 0)
                    return f;
            }
        }
        return f - tmp;
    }

    // 从汇点到源点的路径入队列的顺序
    boolean modlabel() {
        for (int i = 0; i <= n; i++)
            d[i] = INF; // d变量是Cij，即此刻的状态到下一刻状态的费用
        d[des] = 0;
        Deque<Integer> Q = new ArrayDeque<Integer>();
        Q.addLast(des);// (des);
        while (!Q.isEmpty()) {
            int u = Q.getFirst(), tmp;
            Q.pollFirst();

            for (int i = head[u]; i != -1; i = edge[i].next) {
                tmp = d[u] - edge[i].cost;
                if (((edge[edge[i].re].cap) != 0) && (tmp < d[edge[i].v]))// 判断是否满足ZKW算法的条件一
                {
                    d[edge[i].v] = tmp;
                    if ((d[edge[i].v]) <= d[Q.isEmpty() ? src : Q.getFirst()]) {
                        Q.addFirst(edge[i].v);
                    } else {
                        Q.addLast(edge[i].v);
                    }
                }
            }
        }
        // 对走过路径的费用清零
        for (int u = 1; u <= n; u++)
            for (int i = head[u]; i != -1; i = edge[i].next)
                edge[i].cost += d[edge[i].v] - d[u];
        cost += d[src];
        return d[src] < INF; // 如果为真，从汇点到源点找到一条可行路径
    }

    void costflow() {
        while (modlabel()) {
            do {
                Arrays.fill(vis, 0);
            } while (aug(src, INF) == 0);
        }
    }


    String[] findWalk(int[][] weight,
                      boolean[] servers,
                      LinkedList<Vertex<Integer>> clients,
                      int count,
                      int num) {

        int[][] current_weight = new int[weight.length][weight.length];
        int[][] source_weight = new int[weight.length][weight.length];

        int[] path = new int[weight.length];
        boolean[] visited = new boolean[weight.length];
        boolean[] isClient = new boolean[weight.length];
        LinkedList<Integer> stack = new LinkedList<Integer>();
        for (int i = 0; i < weight.length; i++) {
            source_weight[i] = weight[i].clone();
        }

        for (Vertex<Integer> client : clients){
            isClient[client.id] = true;
        }
        LinkedList<String> walks_set = new LinkedList<String>();

        while (modlabel()) {
            do {
                Arrays.fill(vis, 0);
            } while (augNew(src, INF) == 0);
        }

        for(int i = 0; i < count; i ++){
            current_weight[edge[i*2].current-1][edge[i*2].v-1] =
                    source_weight[edge[i*2].current-1][edge[i*2].v-1] - edge[i*2].cap;
        }
        int demand;
        int support;
        for(Vertex<Integer> client : clients){
            demand = client.getDemand();
            while (demand > 0){
                visited = new boolean[weight.length];
                path = new int[weight.length];
                dfs(client.id,
                        current_weight,
                        visited,
                        path,
                        demand,
                        0,
                        servers,
                        isClient,
                        walks_set,
                        num);
                String[] tmp = walks_set.getLast().split(" ");
                support = Integer.parseInt(tmp[tmp.length-1]);
                demand -= support;
            }
        }
//        for(Vertex<Integer> client:clients){
//            for(int i =0; i < weight.length; i++){
//                if(current_weight[client.id][i]!=0)
//                    System.out.println("error");
//            }
//        }
        String[] result = new String[walks_set.size()+2];
        result[0] = ""+walks_set.size();
        result[1] = "";
        for(int i = 0; i < walks_set.size(); i++){
            result[i+2] = walks_set.get(i);
        }
        return result;
    }
    public boolean dfs(int node, int[][] weights, boolean[] visited, int[] path, int demand, int support,
                       boolean[] servers,
                       boolean[] isClient,
                       LinkedList<String> walks_set,
                       int num){

        if(servers[node]){
            String tmp = node +" ";
            weights[path[node]][node] -= support;
            int u;
            for(u = path[node]; !isClient[u]; u = path[u]){
                tmp += u +" ";
                weights[path[u]][u] -= support;
            }
            tmp += (u-num+" "+support);
            walks_set.addLast(tmp);
//            System.out.println(tmp);
            return true;

        }else {
            visited[node] = true;
            for(int i = 0; i < weights.length; i ++){
                if(weights[node][i]!=0 && !visited[i]){
                    path[i] = node;
                    support = Math.min(demand,weights[node][i]);
                    if(dfs(i, weights, visited, path, support, support, servers, isClient, walks_set,num)){
                        visited[node] = false;
                        return true;
                    }
                }
            }
        }
        return true;
    }

    public static String[] getWalk(String[] graphContent, ArrayList<Integer> individual, int[][] weight,
                                   LinkedList<Vertex<Integer>> clients_init) {
        int count = 0;

        ZKW2 zkw = new ZKW2();

        for (int i = 0; i < zkw.MAXM; i++) {
            EDGE2 e = new EDGE2();
            zkw.edge[i] = e;
        }

        zkw.init();

        // 添加边

        String[] nums = graphContent[0].split(" ");
        count = Integer.parseInt(nums[2]) + Integer.parseInt(nums[1]) * 2;
        int num = Integer.parseInt(nums[0]);
        int costOfserver = Integer.parseInt(graphContent[2]);
        zkw.n = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 2;
        zkw.src = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 1;
        zkw.des = zkw.n;
        int demands=0;
        for (int i = 4; i < Integer.parseInt(nums[1]) + 4; i++) {

            String[] data = graphContent[i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + 1, Integer.parseInt(data[1]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
            zkw.add(Integer.parseInt(data[1]) + 1, Integer.parseInt(data[0]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
        }
        for (int i = 0; i < Integer.parseInt(nums[2]); i++) {

            String[] data = graphContent[Integer.parseInt(nums[1]) + 5 + i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[1]) + 1,
                    Integer.parseInt(data[2]), 0);
            demands += Integer.parseInt(data[2]);
        }
        for (int i = 0; i < Integer.parseInt(nums[2]); i++) {

            String[] data = graphContent[Integer.parseInt(nums[1]) + 5 + i].split(" ");

            zkw.add(zkw.src, Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[2]), 0);

        }

        boolean[] servers = new boolean[weight.length];
        for (Integer i : individual) {
            servers[i] = true;
            zkw.add(i + 1, zkw.des, zkw.INF, 0);
        }

        String[] content = zkw.findWalk(weight, servers, clients_init, count, num);
        ArrayList<Integer> useOfServers = new ArrayList<Integer>();
        for (int i : individual) {
            if (zkw.edge[zkw.head[i+1]].cap != zkw.INF) {
                useOfServers.add(i);
            }
        }

//        System.out.println(zkw.ans + costOfserver * (useOfServers.size()));
//        System.out.println(useOfServers);
//        System.out.println(zkw.DEMAND==demands);
        zkw.DEMAND = 0;
//        System.out.println(useOfServers.size());
        return content;

    }

    public static int methodOfzkw1(String[] graphContent, ArrayList<Integer> servers) {
        ZKW2 zkw = new ZKW2();
        // 初始化
        for (int i = 0; i < zkw.MAXM; i++) {
            EDGE2 e = new EDGE2();
            zkw.edge[i] = e;
        }
        zkw.init();
        Arrays.fill(zkw.head, -1);
        zkw.e = 0;
        zkw.ans = zkw.cost = 0;

        // Arrays.fill(zkw.rlx, 0);

        // 添加边
        int demands = 0;
        String[] nums = graphContent[0].split(" ");
        int costOfserver = Integer.parseInt(graphContent[2]);
        zkw.n = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 2;
        zkw.src = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 1;

        zkw.des = zkw.n;

        for (int i = 4; i < Integer.parseInt(nums[1]) + 4; i++) {
            String[] data = graphContent[i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + 1, Integer.parseInt(data[1]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
            zkw.add(Integer.parseInt(data[1]) + 1, Integer.parseInt(data[0]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
        }
        for (int i = 0; i < Integer.parseInt(nums[2]); i++) {

            String[] data = graphContent[Integer.parseInt(nums[1]) + 5 + i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[1]) + 1,
                    Integer.parseInt(data[2]), 0);
            demands += Integer.parseInt(data[2]);
            zkw.add(zkw.src, Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[2]), 0);

        }

        for (int i : servers) {
            zkw.add(i + 1, zkw.des, zkw.INF, 0);
        }

        zkw.costflow();
        ArrayList<Integer> useOfServers = new ArrayList<Integer>();
        ArrayList<Integer> tmp = (ArrayList<Integer>) servers.clone();
        for (Integer i : tmp) {
            if (zkw.edge[zkw.head[i + 1]].cap != zkw.INF) {
                useOfServers.add(i);
            }else {
                servers.remove(i);
            }
        }

        int cost;
        if (demands != zkw.DEMAND) {
            cost = zkw.ans + costOfserver*((demands-zkw.DEMAND)/(zkw.DEMAND/useOfServers.size())+100)*10;
        } else {
            cost = zkw.ans + costOfserver * (useOfServers.size());
        }
        //System.out.println(useOfServers);
        zkw.DEMAND = 0;
        return cost;
    }

    public static ArrayList<Integer> methodOfzkw(String[] graphContent, ArrayList<Integer> servers) {

        int min = methodOfzkw1(graphContent, servers);
        int s = -1;
        for (int i = 0; i < servers.size(); i++) {
            int temp = servers.get(i);
            servers.remove(i);
            // long start = System.currentTimeMillis();

            int t = methodOfzkw1(graphContent, servers);
            if (t == -1) {
                servers.add(i, temp);
                continue;
            }
            // min=Math.min(t, min);

            if (t < min) {
                min = t;
                s = i;

            }
            servers.add(i, temp);
        }

        if (s != -1) {
            servers.remove(s);
        }
        return servers;

    }

    public static int methodOfzkwNew(String[] graphContent,ArrayList<Integer> servers) {

        ZKW2 zkw = new ZKW2();

        // 初始化
        for (int i = 0; i < zkw.MAXM; i++) {
            EDGE2 e = new EDGE2();
            zkw.edge[i] = e;
        }
        zkw.init();
        Arrays.fill(zkw.head, -1);
        zkw.e = 0;
        zkw.ans = zkw.cost = 0;


        // 添加边
        int demands=0;
        String[] nums = graphContent[0].split(" ");
        int costOfserver=Integer.parseInt(graphContent[2]);
        zkw.n = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 2;
        zkw.src = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 1;

        zkw.des = zkw.n;

        for (int i = 4; i < Integer.parseInt(nums[1]) + 4; i++) {
            String[] data = graphContent[i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + 1, Integer.parseInt(data[1]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
            zkw.add(Integer.parseInt(data[1]) + 1, Integer.parseInt(data[0]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
        }
        for (int i = 0; i < Integer.parseInt(nums[2]); i++) {

            String[] data = graphContent[Integer.parseInt(nums[1]) + 5 + i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[1]) + 1,
                    Integer.parseInt(data[2]), 0);
            demands+=Integer.parseInt(data[2]);
            zkw.add(zkw.src, Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[2]), 0);

        }

        for(int i:servers){
            zkw.add(i+1,zkw.des, zkw.INF, 0);
        }

        zkw.costflow();
        ArrayList<Integer> useOfServers=new ArrayList<Integer>();
        for(int i:servers){
            if(zkw.edge[zkw.head[i+1]].cap!=zkw.INF){;
                useOfServers.add(i);
            }
        }
        int cost;
        if(demands != zkw.DEMAND){
            cost = -1;
        }else {
            cost = zkw.ans+costOfserver*(useOfServers.size());
        }
        zkw.DEMAND = 0;
        return cost;
    }

    public static Result methodOfzkw2(String[] graphContent, ArrayList<Integer> servers) {
        ZKW2 zkw = new ZKW2();
        // 初始化
        for (int i = 0; i < zkw.MAXM; i++) {
            EDGE2 e = new EDGE2();
            zkw.edge[i] = e;
        }
        zkw.init();
        Arrays.fill(zkw.head, -1);
        zkw.e = 0;
        zkw.ans = zkw.cost = 0;

        // Arrays.fill(zkw.rlx, 0);

        // 添加边
        int demands = 0;
        String[] nums = graphContent[0].split(" ");
        int costOfserver = Integer.parseInt(graphContent[2]);
        zkw.n = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 2;
        zkw.src = Integer.parseInt(nums[2]) + Integer.parseInt(nums[0]) + 1;

        zkw.des = zkw.n;

        for (int i = 4; i < Integer.parseInt(nums[1]) + 4; i++) {
            String[] data = graphContent[i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + 1, Integer.parseInt(data[1]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
            zkw.add(Integer.parseInt(data[1]) + 1, Integer.parseInt(data[0]) + 1, Integer.parseInt(data[2]),
                    Integer.parseInt(data[3]));
        }
        for (int i = 0; i < Integer.parseInt(nums[2]); i++) {

            String[] data = graphContent[Integer.parseInt(nums[1]) + 5 + i].split(" ");

            zkw.add(Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[1]) + 1,
                    Integer.parseInt(data[2]), 0);
            demands += Integer.parseInt(data[2]);
            zkw.add(zkw.src, Integer.parseInt(data[0]) + Integer.parseInt(nums[0]) + 1, Integer.parseInt(data[2]), 0);

        }

        for (int i : servers) {
            zkw.add(i + 1, zkw.des, zkw.INF, 0);
        }

        zkw.costflow();
        ArrayList<Integer> useOfServers = new ArrayList<Integer>();
        ArrayList<Integer> tmp = (ArrayList<Integer>) servers.clone();
        for (Integer i : tmp) {
            if (zkw.edge[zkw.head[i + 1]].cap != zkw.INF) {
                useOfServers.add(i);
            }else {
                servers.remove(i);
            }
        }

        int cost;

        if (demands != zkw.DEMAND) {
            cost = zkw.ans + costOfserver * (useOfServers.size());
            return new Result(false,zkw.DEMAND,cost,demands);
        } else {
            cost = zkw.ans + costOfserver * (useOfServers.size());
            return new Result(true,demands,cost,zkw.DEMAND);
        }
    }

}

class Result{
    boolean flag;
    int support;
    int cost;
    int demand;
    public Result(boolean flag, int support, int cost, int demand){
        this.flag = flag;
        this.support = support;
        this.cost = cost;
        this.demand = demand;
    }
}










