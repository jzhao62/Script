

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;



public class Lab2 {
	public static ArrayList<sequence> allSequence = new ArrayList<sequence>();
	public static ArrayList<sequence> tuneSequence = new ArrayList<sequence>();
	public static ArrayList<sequence> testSequence = new ArrayList<sequence>();


	public static ArrayList<ArrayList<Integer>> trainInputUnit = new ArrayList<ArrayList<Integer>>();
	public static double[][]traininput;

	public static ArrayList<double[]> trainoutputUnit = new ArrayList<double[]>();
	public static double [][] trainexpectedOutPut;
	//------------------------------------------------------------

	public static ArrayList<ArrayList<Integer>> tuneInputUnit = new ArrayList<ArrayList<Integer>>();
	public static double[][]tuneinput;

	public static ArrayList<double[]> tuneoutputUnit = new ArrayList<double[]>();
	public static double [][] tuneExpectedOutPut;
	//------------------------------------------------------------

	public static ArrayList<ArrayList<Integer>> testInputUnit = new ArrayList<ArrayList<Integer>>();
	public static double[][]testinput;

	public static ArrayList<double[]> testoutputUnit = new ArrayList<double[]>();
	public static double [][] testExpectedOutPut;
	//------------------------------------------------------------



	public static double[][] dummyOutPut;
	public static HashMap<String, double[]> stringConversion;
	public static ArrayList<String> num;



	public static void main (String args[]) throws IOException{



		// args[0] is the name of input file, please include the .txt
		//------------------------------------------------------------

		parse(args[0]);
		stringConversion = StringConversion(num);
		int[]tuneIndexing = tuneIndexArray();
		initializeTuneTest(tuneIndexing);
		Collections.shuffle(tuneSequence);
		Collections.shuffle(testSequence);



		System.out.println("Finishing initializing 3 sets");
		//------------------------------------------------------------

		finalizeTrainData();
		finalizetestData();
		finalizeTuneData();


		// call the neural network and build it
		//------------------------------------------------------------
		int Runs;


		//------------------------------------------------------------to TA------------------------------------------------------------

		// You can manually set the max epochs here (in this case, i obtained 300 epochs results after running a whole night
		// you can tune it and the program will output the best test set result based on your epoch setting 
		int max = 1000;




		ArrayList<Double> trainResult = new ArrayList<Double>();
		ArrayList<Double> tuneResult = new ArrayList<Double>();
		ArrayList<Double> testResult = new ArrayList<Double>();
		ArrayList<Double> SquaredError = new ArrayList<Double>();

		double[][] tuneTest = new double[max][2];




		for(Runs = 1; Runs <= max; Runs+=1){




			NeuralNetwork ann = new NeuralNetwork(20*17,10,3);
			double minErrorCondition = 0.001;

			double[] trainAccuracy = ann.run(Runs, minErrorCondition,true);

			NeuralNetwork trainedShell = new NeuralNetwork(ann.getInputLayer(),ann.getHiddenLayer(),ann.getOutPutLayer());
			NeuralNetwork trainedShell2 = new NeuralNetwork(ann.getInputLayer(),ann.getHiddenLayer(),ann.getOutPutLayer());

			double tuneAccuracy = trainedShell.testANN(tuneinput, tuneExpectedOutPut);
			//		System.out.print("Test:");
			double testAccuracy = trainedShell2.testANN(testinput, testExpectedOutPut);

			tuneTest[Runs-1][0] = tuneAccuracy;
			tuneTest[Runs-1][1] = testAccuracy;

			trainResult.add(trainAccuracy[0]);
			SquaredError.add(trainAccuracy[1]);
			tuneResult.add(tuneAccuracy);
			testResult.add(testAccuracy);





			System.out.println(trainAccuracy[0] + " Tune: " + tuneAccuracy + " Test: " + testAccuracy);

			System.out.println("##### EPOCH " + Runs +"\n");


			try(Writer writer1 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("trainAccuracy.txt"), StandardCharsets.UTF_8))) 
			{
				for(double a : trainResult){
					writer1.write(String.valueOf(a) + "\n");

				}
			}
			try(Writer writer2 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("TuneAccuracy.txt"), StandardCharsets.UTF_8))) 
			{
				for(double a :tuneResult){
					writer2.write(String.valueOf(a) + "\n");

				}
			}

			try(Writer writer3 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("testAccuracy.txt"), StandardCharsets.UTF_8))) 
			{
				for(double a : testResult){
					writer3.write(String.valueOf(a) + "\n");

				}
			}

			try(Writer writer4 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("SquaredError.txt"), StandardCharsets.UTF_8))) 
			{
				for(double a : SquaredError){
					writer4.write(String.valueOf(a) + "\n");

				}
			}


			catch (IOException e) {
				e.printStackTrace();
			}
		}

		int index = findMax(tuneTest);
		System.out.println("Highest testResult: " + (float)tuneTest[index][1]*100 + "%" + " when tuneResult: "
				+ (float)tuneTest[index][0]*100 + "%" + " at "
				+ " ##Epoch: " + index);







	}


	static int[] tuneIndexArray(){
		//	int[] temp = new int[25];
		int[]temp = new int[4];
		int counter = 0;
		for(int i = 0; i < temp.length; i ++){
			//	counter = counter + 5;
			counter = counter + 2;
			temp[i] = counter;
		}

		return temp;


	}
	static int findMax(double[][]array){
		int index = 0;
		double max = 0;
		for(int i = 0; i < array.length;i++){
			if(array[i][0]>max){
				index = i;
				max = array[i][0];
			}

		}

		return index;
	}





	static void finalizetestData(){
		sequence shell = new sequence();

		for(sequence b : testSequence){

			ArrayList<ArrayList<basePiece>> inputProtein = b.generateInputSeries(17);

			for(ArrayList<basePiece> a : inputProtein){

				//		shell.displaySequence(a);


				ArrayList<Integer> temp =	shell.convertToBinaryForm(a, 20, stringConversion);
				testInputUnit.add(temp);
				testoutputUnit.add(shell.getProteinOutPut(a));

			}

		}

		testinput = inputArrayListConversion(testInputUnit,17);
		testExpectedOutPut = outputConversion(testoutputUnit,17,trainInputUnit);
		dummyOutPut = initializeDummyOutPut();
	}
	//------------------------------------------------------------

	static void finalizeTuneData(){
		sequence shell = new sequence();

		for(sequence b : tuneSequence){

			ArrayList<ArrayList<basePiece>> inputProtein = b.generateInputSeries(17);

			for(ArrayList<basePiece> a : inputProtein){

				//		shell.displaySequence(a);


				ArrayList<Integer> temp =	shell.convertToBinaryForm(a, 20, stringConversion);
				tuneInputUnit.add(temp);
				tuneoutputUnit.add(shell.getProteinOutPut(a));

			}

		}

		tuneinput = inputArrayListConversion(tuneInputUnit,17);
		tuneExpectedOutPut = outputConversion(tuneoutputUnit,17,tuneInputUnit);
		dummyOutPut = initializeDummyOutPut();

	}	
	//------------------------------------------------------------


	static void finalizeTrainData(){
		sequence shell = new sequence();

		for(sequence b : allSequence){

			ArrayList<ArrayList<basePiece>> inputProtein = b.generateInputSeries(17);

			for(ArrayList<basePiece> a : inputProtein){

				//		shell.displaySequence(a);


				ArrayList<Integer> temp =	shell.convertToBinaryForm(a, 20, stringConversion);
				trainInputUnit.add(temp);
				trainoutputUnit.add(shell.getProteinOutPut(a));

			}

		}

		traininput = inputArrayListConversion(trainInputUnit,17);
		trainexpectedOutPut = outputConversion(trainoutputUnit,17,trainInputUnit);
		dummyOutPut = initializeDummyOutPut();
	}
	//------------------------------------------------------------



	static void testANN(double[][]input, double[][]expectedOutPut){


	}





	static void initializeTuneTest(int[] tuneIndexArray){

		for(int j = tuneIndexArray.length-1; j >=0; j--){
			sequence test = allSequence.remove(tuneIndexArray[j]);
			sequence tune = allSequence.remove(tuneIndexArray[j]-1);
			tuneSequence.add(tune);
			testSequence.add(test);

		}


	}




















	public static void parse(String fileName) throws FileNotFoundException{
		num = new ArrayList<String>();
		File file = new File(fileName);
		Scanner sc = new Scanner(file);
		String input = sc.next();
		while (sc.hasNextLine()) 
		{
			//build one sequence from "<>" to <end>
			if(input.contains("<>")){
				sequence protein = new sequence();
				while(!input.contains("end")){

					input = sc.nextLine();
					if(input.contains("end")){
						break;
					}
					else if(input.contains("<>")){
						break;

					}
					String[] result = input.split(" ");
					basePiece a = new basePiece(result[0], result[1]);
					if(!num.contains(result[0])){
						num.add(result[0]);
					}
					protein.addBasePair(a);
				}

				allSequence.add(protein);	
			}
			else{
				input = sc.nextLine();

			}
		}

		Collections.sort(num);
	}

	public static HashMap<String, double[]> StringConversion(ArrayList<String> num){
		stringConversion = new HashMap<String,double[]>();
		int key = 0;
		for(String a : num){
			double[]temp = new double[20];
			for(int i = 0; i <temp.length;i++){
				temp[i] = 0;
			}
			temp[key] = 1;
			stringConversion.put(a, temp);
			key++;
		}
		return stringConversion;

	}


	public sequence getSequence(int i, ArrayList<sequence> input){

		return input.get(i);
	}



	public static double[][] inputArrayListConversion(ArrayList<ArrayList<Integer>> inputAA, int window17){


		double[][]input = new double[inputAA.size()][21*window17];
		int key = 0;

		for(ArrayList<Integer> a : inputAA){
			double[]oneInput = new double [21*window17];
			for(int i = 0; i < a.size();i++){
				oneInput[i] = a.get(i);

			}
			input[key] = oneInput;	
			key ++;

		}

		return input;


	}


	public static double[][] outputConversion(ArrayList<double[]>singleSequenceOutPut, int window17, ArrayList<ArrayList<Integer>>trainInputUnit){

		double[][]temp = new double[trainInputUnit.size()][window17];

		int key = 0;
		for(double[] a : singleSequenceOutPut){
			temp[key] = a;
			key++;
		}

		return temp;

	}

	public static double[][] initializeDummyOutPut(){
		dummyOutPut = new double[trainexpectedOutPut.length][1];

		for(int i = 0; i < trainexpectedOutPut.length; i++){
			dummyOutPut[i][0] = -1;
		}

		return dummyOutPut;

	}




}


class basePiece{
	public String aAcids;
	public String label;

	public basePiece(String input, String label){
		this.aAcids = input;
		this.label = label;

	}

	public String getLabel(){
		return this.label;
	}
	public String getAcid(){
		return this.aAcids;
	}



}
/**Represent one sequence starting from <> to <end>, and crop the entire sequence into multiple 
 * input units that satisfy the window size.
 * 
 * 
 * @author jingyizhao
 *
 */

class sequence extends ArrayList<basePiece>{

	public ArrayList<basePiece> proteinStructure = new ArrayList<basePiece> ();
	public ArrayList<ArrayList<basePiece>> inputSeries = new ArrayList<ArrayList<basePiece>>();
	public int length;


	public sequence (){

	}

	public void addBasePair(basePiece a){
		this.proteinStructure.add(a);
		length++;
	}

	public ArrayList<ArrayList<basePiece>> generateInputSeries(int window17){

		ArrayList<ArrayList<basePiece>> core = new ArrayList<ArrayList<basePiece>>();
		basePiece[] protein = this.proteinStructure.toArray(new basePiece[this.proteinStructure.size()]);
		int start = 0;
		int end = 0+ window17;

		while(end <= this.proteinStructure.size()){
			basePiece[] newSubStructure = new basePiece[window17];
			for(int i = 0; i < window17; i ++){
				newSubStructure[i] = protein[i+start];
			}

			start++;
			end++;
			ArrayList<basePiece> temp = new ArrayList<basePiece>(Arrays.asList(newSubStructure));
			core.add(temp);
		}

		this.inputSeries = core;
		return this.inputSeries;
	}




	public void displaySequence(ArrayList<basePiece> proteinStructure){
		for(basePiece a: proteinStructure){
			System.out.println(a.getAcid());
		}
		System.out.println("______________");

	}


	public ArrayList<basePiece> getProteinStructure(){
		return this.proteinStructure;
	}

	// return the output label for a 17 unit protein.located in the median. require the input size to be odd;

	public double[] getProteinOutPut(ArrayList<basePiece> input){

		double[]outputLabel = new double[3];
		for(int i = 0; i < outputLabel.length;i++){
			outputLabel[i] = 0;
		}

		String result = input.get((input.size()-1)/2).getLabel();
		if(result.contains("e")){
			outputLabel[0] = 1;
		}
		else if(result.contains("h")){
			outputLabel[1] = 1;
		}
		else if (result.contains("_")){
			outputLabel[2] = 1;
		}
		return outputLabel;



	}

	// Expected to be a 20 x 17 length
	/**
	 * Output a int[20] x 17 size array for each basePiece, and then combine them to form a total of 20x17 input Unit
	 * @param singleInput: represents a single 17 unit protein
	 * @param length: 20
	 * @param windowSize: 17
	 * @return
	 */
	public ArrayList<Integer> convertToBinaryForm(ArrayList<basePiece> singleInput, int length, HashMap<String, double[]>map){

		ArrayList<Integer> inputUnit = new ArrayList<Integer>();


		for(basePiece a : singleInput){
			//			// Initialize every input cell into 0; 
			//			for(int i = 0; i < sInput.length;i++){
			//				sInput[i] = 0;
			//			}
			//			//if basePiece is A, corresponding to 100000.....0;
			//			char change = a.getAcid().charAt(0);
			//			//	System.out.println((int)change - 65 +" "+sInput.length);
			//			sInput[(int) change-65] = 1;

			double[]binaryValue = map.get(a.getAcid());


			for(int i = 0; i < binaryValue.length;i++){
				inputUnit.add((int)binaryValue[i]);
			}


		}

		return inputUnit;



	}




}

class Connection {
	double weight = 0;
	double prevDeltaWeight = 0; // for momentum
	double deltaWeight = 0;

	final Neuron leftNeuron;
	final Neuron rightNeuron;
	static int counter = 0;
	final public int id; // auto increment, starts at 0

	public Connection(Neuron fromN, Neuron toN) {
		leftNeuron = fromN;
		rightNeuron = toN;
		id = counter;
		counter++;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double w) {
		weight = w;
	}

	public void setDeltaWeight(double w) {
		prevDeltaWeight = deltaWeight;
		deltaWeight = w;
	}

	public double getPrevDeltaWeight() {
		return prevDeltaWeight;
	}

	public Neuron getFromNeuron() {
		return leftNeuron;
	}

	public Neuron getToNeuron() {
		return rightNeuron;
	}
}








class Neuron {   
	static int counter = 0;
	final public int id;  // auto increment, starts at 0

	// initialize the bias 
	Connection biasConnection;
	final double bias = -1;
	double output;


	// contains the connections between the current Neuron and all other inputs that are attached to it.
	ArrayList<Connection> Inconnections = new ArrayList<Connection>();

	// Easy to look up connections 
	HashMap<Integer,Connection> connectionLookup = new HashMap<Integer,Connection>();

	public Neuron(){        
		id = counter;
		counter++;
	}

	/**
	 * Compute Sj = Wij*Aij + w0j*bias, which is the output for a single Hidden Unit
	 */
	public void calculateOutput(){
		double s = 0;
		// Iterate over all connections
		for(Connection con : Inconnections){
			Neuron leftNeuron = con.getFromNeuron();
			double weight = con.getWeight();
			double a = leftNeuron.getOutput(); //output from previous layer

			s = s + (weight*a);
		}
		s = s + (biasConnection.getWeight()*bias);

		output = g(s);
	}


	/**
	 * Activation function 
	 * @param input 
	 * @return output from activation function 
	 */
	double g(double x) {
		return sigmoid(x);
	}
	/**
	 * Activation function as a sigmoid function 
	 * @param x
	 * @return
	 */
	double sigmoid(double x) {
		return 1.0 / (1.0 +  (Math.exp(-x)));
	}

	/**connect the current hidden unit to a list of Neurons
	 * 
	 * @param inNeurons
	 */
	public void addInConnectionsS(ArrayList<Neuron> inNeurons){
		for(Neuron n: inNeurons){
			Connection con = new Connection(n,this);
			Inconnections.add(con);
			connectionLookup.put(n.id, con);
		}
	}

	/**
	 * retrieve a certain connection given an index
	 * 
	 * @param neuronIndex
	 * @return a Connection
	 */
	public Connection getConnection(int neuronIndex){
		return connectionLookup.get(neuronIndex);
	}


	public void addInConnection(Connection con){
		Inconnections.add(con);
	}
	public void addBiasConnection(Neuron n){
		Connection con = new Connection(n,this);
		biasConnection = con;
		Inconnections.add(con);
	}
	public ArrayList<Connection> getAllInConnections(){
		return Inconnections;
	}

	public double getBias() {
		return bias;
	}
	public double getOutput() {
		return output;
	}

	/**
	 * S
	 * 
	 * @param o
	 */
	public void setOutput(double o){
		output = o;
	}
}




class NeuralNetwork {



	boolean isTrained = false;
	DecimalFormat df;
	final Random rand = new Random();
	ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
	ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
	ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
	Neuron bias = new Neuron();
	int[] layers;
	final int randomWeightMultiplier = 1;
	final double lamda = 0.001;

	final double epsilon = 0.00000000001;

	final double learningRate = 0.25;
	final double momentum = 0.7;

	//------------------------------------------------------------
	double inputs[][] = Lab2.traininput;
	double expectedOutputs[][] = Lab2.trainexpectedOutPut;
	//------------------------------------------------------------


	double resultOutputs[][] = Lab2.dummyOutPut; 
	double output[];

	// for weight update all
	HashMap<String, Double> weightUpdate = new HashMap<String, Double>();



	// Construct a empty Neural Network


	public NeuralNetwork(ArrayList<Neuron> inputLayer,ArrayList<Neuron> hiddenLayer,ArrayList<Neuron> outputLayer){
		this.inputLayer = inputLayer;
		this.hiddenLayer= hiddenLayer;
		this.outputLayer = outputLayer;

	}



	// Construct a Neural Network with given the size of inputs, hidden units, and output units
	//------------------------------------------------------
	public NeuralNetwork(int input, int hidden, int output) {
		this.layers = new int[] { input, hidden, output };
		df = new DecimalFormat("#.0#");

		/**
		 * Create all neurons and connections Connections are created in the
		 * neuron class
		 */
		for (int i = 0; i < layers.length; i++) {
			if (i == 0) { // input layer
				for (int j = 0; j < layers[i]; j++) {
					Neuron neuron = new Neuron();
					inputLayer.add(neuron);
				}


			} else if (i == 1) { // hidden layer
				for (int j = 0; j < layers[i]; j++) {
					Neuron neuron = new Neuron();
					neuron.addInConnectionsS(inputLayer);
					neuron.addBiasConnection(bias);
					hiddenLayer.add(neuron);
				}
			}

			else if (i == 2) { // output layer
				for (int j = 0; j < layers[i]; j++) {
					Neuron neuron = new Neuron();
					neuron.addInConnectionsS(hiddenLayer);
					neuron.addBiasConnection(bias);
					outputLayer.add(neuron);
				}
			} 
			else {
				System.out.println("!Error NeuralNetwork init");
			}
		}

		// initialize random weights
		for (Neuron neuron : hiddenLayer) {
			ArrayList<Connection> connections = neuron.getAllInConnections();
			for (Connection conn : connections) {
				double newWeight = getRandom();
				conn.setWeight(newWeight);
			}
		}
		for (Neuron neuron : outputLayer) {
			ArrayList<Connection> connections = neuron.getAllInConnections();
			for (Connection conn : connections) {
				double newWeight = getRandom();
				conn.setWeight(newWeight);
			}
		}

		// reset id counters
		Neuron.counter = 0;
		Connection.counter = 0;

		//        if (isTrained) {
		//            trainedWeights();
		//            updateAllWeights();
		//        }
	}




	ArrayList<Neuron> getInputLayer(){
		return this.inputLayer;
	}

	ArrayList<Neuron> getHiddenLayer(){
		return this.hiddenLayer;
	}

	ArrayList<Neuron> getOutPutLayer(){
		return this.outputLayer;
	}

	// Random value generator 
	double getRandom() {
		return randomWeightMultiplier * (rand.nextDouble() * 2 - 1);

	}



	/**
	 * INITIALIZE THE OUTPUT OF THE INPUT LAYER
	 * @param inputs
	 *            There is equally many neurons in the input layer as there are
	 *            in input variables
	 */
	public void setInput(double inputs[]) {
		for (int i = 0; i < inputLayer.size(); i++) {
			inputLayer.get(i).setOutput(inputs[i]);
		}
	}



	public double[] getOutput() {
		double[] outputs = new double[outputLayer.size()];
		for (int i = 0; i < outputLayer.size(); i++)
			outputs[i] = outputLayer.get(i).getOutput();
		return outputs;
	}



	/**
	 * Calculate the output of the neural network based on the input during The forward
	 * operation
	 */
	public void activate() {

		for (Neuron n : hiddenLayer)
			n.calculateOutput();
		for (Neuron n : outputLayer)
			n.calculateOutput();
	}

	/**
	 * UPDATE THE WEIGHT ON ALL CONNECTIONS FOR ONE TIME. THE METHOD IS VOID IN THAT THE CONNECTION IS UPDATED IN THE FIELD
	 * employed both momentum and weight decay
	 * 
	 * @param expectedOutput
	 *            first calculate the partial derivative of the error with
	 *            respect to each of the weight leading into the output neurons
	 *            bias is also updated here
	 */
	public void applyBackpropagation(double expectedOutput[]) {

		// error check, normalize value ]0;1[

		for (int i = 0; i < expectedOutput.length; i++) {
			double d = expectedOutput[i];
			if (d < 0 || d > 1) {
				if (d < 0)
					expectedOutput[i] = 0 + epsilon;
				else
					expectedOutput[i] = 1 - epsilon;
			}
		}

		// THE WEIGHT OF THE CONNECTION LINKED TO THE
		// OUTPUT UNIT IS UPDATED ONCE

		int i = 0;
		for (Neuron n : outputLayer) {
			//GET ALL INCOMING CONNECTIONS FOR A GIVEN OUTPUT NEURON
			ArrayList<Connection> connections = n.getAllInConnections();

			//FOR EACH CONNECTION
			for (Connection con : connections) {

				//OBTAIN THE NET OUTPUT FOR A GIVEN OUTPUT UNIT
				double ak = n.getOutput();

				//OBTAIN THE NET OUTPUT FOR A GIVEN INCOMING UNIT
				double ai = con.leftNeuron.getOutput();

				//OBTAIN THE DESIRED OUTPUT 
				double desiredOutput = expectedOutput[i];

				double partialDerivative = -ak * (1 - ak) * ai
						* (desiredOutput - ak);
				double deltaWeight = -learningRate * partialDerivative;
				double newWeight = con.getWeight() + deltaWeight - lamda*learningRate*con.getWeight();
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
			i++;
		}


		// THE WEIGHT OF THE CONNECTION LINKED TO THE
		// HIDDEN UNIT IS UPDATED ONCE

		for (Neuron n : hiddenLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				double aj = n.getOutput();
				double ai = con.leftNeuron.getOutput();
				double sumKoutputs = 0;
				int j = 0;
				for (Neuron out_neu : outputLayer) {
					double wjk = out_neu.getConnection(n.id).getWeight();
					double desiredOutput = (double) expectedOutput[j];
					double ak = out_neu.getOutput();
					j++;
					sumKoutputs = sumKoutputs
							+ (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
				}

				double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
				double deltaWeight = -learningRate * partialDerivative;
				double newWeight = con.getWeight() + deltaWeight - lamda*learningRate*con.getWeight();
				con.setDeltaWeight(deltaWeight);
				con.setWeight(newWeight + momentum * con.getPrevDeltaWeight());
			}
		}
	}







	/**
	 * // Train neural network until minError reached or maxSteps exceeded
	 * 
	 * the first returned element is training accuracy after certain epoch
	 * the second return element is squareErrory after epoch
	 * 
	 * @param maxSteps
	 * @param minError
	 */
	double[] run(int maxSteps, double minError, boolean onDisplay) {

		double[] temp = new double[2];
		int i;

		double error = 1;



		// Each time a new epoch happens, the backpropagation is applied one time more, until the criteria is met.     
		for (i = 0; i < maxSteps && error > minError; i++) {
			error = 0;
			for (int p = 0; p < inputs.length; p++) {

				// ONE INPUT IS ADDED IN EACH TIME 
				setInput(inputs[p]);

				activate();

				output = getOutput();
				resultOutputs[p] = output;

				for (int j = 0; j < expectedOutputs[p].length; j++) {
					double err = Math.pow(output[j] - expectedOutputs[p][j], 2);
					error += err;
					//System.out.println(error);

				}

				applyBackpropagation(expectedOutputs[p]);
			}

		}

		//   printResult();			
		System.out.println("Sum of squared errors during training = " + error);

		temp[0] = printAccuracy(this.inputs,this.resultOutputs,this.expectedOutputs);
		temp[1] = error;

		return temp;


	}

	double testANN(double[][]input,double[][]expectedOutPut){
		for(int p = 0; p < input.length;p++){

			setInput(input[p]);
			activate();
			output = getOutput();
			resultOutputs[p] = output;
		}




		return printAccuracy(input,this.resultOutputs,expectedOutPut);




	}


	void printResult()
	{
		double counter = 0;
		double correct = 0;
		for (int p = 0; p < inputs.length; p++) {




			System.out.print("INPUTS: ");
			for (int x = 0; x < layers[0]; x++) {
				System.out.print((int)inputs[p][x]);
			}

			System.out.print(" EXPECTED: ");
			for (int x = 0; x < layers[2]; x++) {
				System.out.print((int)expectedOutputs[p][x] + " ");
			}

			System.out.print("ACTUAL: ");


			int index = largestValueIndex(resultOutputs[p]);

			if((int)expectedOutputs[p][index]==1){
				correct ++;
			}


			for (int x = 0; x < layers[2]; x++) {


				System.out.print(resultOutputs[p][x] + " ");
			}

			counter++;
			System.out.println();
		}

		System.out.println(counter + " " + correct);
		System.out.println();
	}

	double printAccuracy(double[][]input, double[][]resultOutPut, double[][]expectedOutPut){


		double counter = 0;
		double correct = 0;
		for (int p = 0; p < inputs.length; p++) {

			int index = largestValueIndex(resultOutputs[p]);

			if((int)expectedOutputs[p][index]==1){
				correct ++;
			}

			counter++;

		}   

		return  (float)correct/counter;


	}









	// return the neural NetID and InConnection in a single string 

	String weightKey(int neuronId, int conId) {
		return "N" + neuronId + "_C" + conId;
	}

	/**
	 * Take from hash table and put into all weights
	 */
	public void updateAllWeights() {
		// update weights for the output layer
		for (Neuron n : outputLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				String key = weightKey(n.id, con.id);
				double newWeight = weightUpdate.get(key);
				con.setWeight(newWeight);
			}
		}
		// update weights for the hidden layer
		for (Neuron n : hiddenLayer) {
			ArrayList<Connection> connections = n.getAllInConnections();
			for (Connection con : connections) {
				String key = weightKey(n.id, con.id);
				double newWeight = weightUpdate.get(key);
				con.setWeight(newWeight);
			}
		}
	}


	public int largestValueIndex(double[] input){
		double max = -1;
		int index = -1;

		for(int i = 0; i < input.length; i++){
			if(input[i] > max){
				index = i;
				max = input[i];
			}
		}

		return index;

	}
}









