package project2;

import project2.algorithms.*;
import project2.entity.*;
import project2.utility.*;
import project2.agents.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
// import java.util.function.BiFunction;
// import java.util.function.Predicate;


public class Simulator {
    /**
     * Main Execution Method
     *
     * @param args Input arguments X - Grid X size, Y - Entity.Grid Y size,
     *             Iterations - Number of Iterations (defaults to 100)
     */
    public static void main(String[] args) {
        int x = Integer.parseInt(args[0]);
        int y = Integer.parseInt(args[1]);
        int iterations = args.length > 2 ? Integer.parseInt(args[2]): 100;
        generateData(x, y, iterations, 30);
    }

    public static void generateData(int xDim, int yDim, int iterations, int prob) {
        SearchAlgo algo = new AStarSearch(Heuristics::manhattanDistance);
        Point start = new Point(0, 0);
        Point end = new Point(xDim-1, yDim-1);
        InferenceAgent agent = new BetterInferenceAgent();

        ArrayList<GridWorldInfo> results = new ArrayList<>();
        for(int i = 0; i < iterations; i++) {
            Grid world = getSolvableMaze(xDim, yDim, algo, prob);
            printMaze(world);
            Robot rob = new Robot(start, end, agent, world, algo, false);
            results.add(rob.run());
        }

        printResultsToCsv("p2-realagent.csv", results);
    }

    public static void printMaze(Grid world) {
        for(int y = 0; y < world.getYSize(); y++) {
            for(int x = 0; x < world.getXSize(); x++) {
                Point pt = new Point(x, y);
                GridCell cell = world.getCell(pt);
                if (cell.isBlocked()) {
                    System.out.print("1 ");
                } else {
                    System.out.print("0 ");
                }
            }
        }
    }

    /**
     * This method returns a solvable maze for given inputs. It uses a search algo
     * to make sure there is a path to the end.
     *
     * @param xDim Dimension of the Entity.Grid's X coord
     * @param yDim Dimension of the Entity.Grid's Y coord
     * @param algo Algorithm to use for searching
     * @param prob Probability of a space being blocked
     * @return returns a solvable maze
     */
    public static Grid getSolvableMaze(int xDim, int yDim, SearchAlgo algo, int prob) {
        Point start = new Point(0, 0);
        Point end = new Point(xDim - 1, yDim - 1);
        GridWorldInfo completeResult;
        Grid grid;

        do {
            grid = new Grid(xDim, yDim, prob);
            completeResult = algo.search(start, end, grid, cell -> cell.isBlocked());
        } while (completeResult.path == null);

        return grid;
    }

    /**
     * Takes a list of Entity.GridWorldInfo: {@link GridWorldInfo} and prints it to
     * a pre-designed csv template
     *
     * @param fileName      name of the file
     * @param gridWorldInfo List of GridWorldInfos
     */
    public static void printResultsToCsv(String fileName, List<GridWorldInfo> gridWorldInfo) {
        try (PrintWriter writer = new PrintWriter(new File(fileName))) {

            StringBuilder sb = new StringBuilder();
            sb.append("Probability");
            sb.append(',');
            sb.append("Solvable");
            sb.append(',');
            sb.append("Runtime");
            sb.append(',');
            sb.append("Path Length");
            sb.append(',');
            sb.append("Number of Cells Processed");
            sb.append(',');
            sb.append("Number of Bumps");
            sb.append(',');
            sb.append("Number of Planning Steps");
            sb.append(',');
            sb.append("Number of Cells Determined");
            sb.append('\n');
            writer.write(sb.toString());

            for (GridWorldInfo info : gridWorldInfo) {
                sb = new StringBuilder();
                sb.append(info.probability);
                sb.append(',');
                sb.append(!Double.isNaN(info.trajectoryLength));
                sb.append(',');
                sb.append(info.runtime);
                sb.append(',');
                sb.append(info.trajectoryLength);
                sb.append(',');
                sb.append(info.numberOfCellsProcessed);
                sb.append(',');
                sb.append(info.numBumps);
                sb.append(',');
                sb.append(info.numPlans);
                sb.append(',');
                sb.append(info.numCellsDetermined);
                sb.append('\n');
                writer.write(sb.toString());
            }
        } catch (FileNotFoundException e) {
            System.err.println(e.getMessage());
        }
    }

}
