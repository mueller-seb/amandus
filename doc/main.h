/**
 * \mainpage
 *
 *\htmlonly
 * <div style="float:left"><a title="By Unk (http://www.math.ru/history/people/Schwarz)
 * [Public domain], via Wikimedia Commons"
 * href="https://commons.wikimedia.org/wiki/File%3AKarl_Hermann_Amandus_Schwarz.jpg"><img
 * width="256" alt="Karl Hermann Amandus Schwarz"
 * src="https://upload.wikimedia.org/wikipedia/commons/e/ea/Karl_Hermann_Amandus_Schwarz.jpg"/></a></div>
 * \endhtmlonly
 *
 * Amandus is a platform for solving mixed element problems based on
 * the software library <a
 * href="https://www.dealii.org/">deal.II</a>. It is named after <a
 * href="https://en.wikipedia.org/wiki/Hermann_Schwarz">Hermann
 * Amandus Schwarz</a>, who invented the alternating Schwarz method
 * and whose name is commemorated in the inequality named after him,
 * Bunyakovsky, and Cauchy in varying combinations.
 *
 * The purpose of Amandus is enabling the solution of PDE problems
 * without much prior knowledge of C++ or deal.II. To this end,
 * Amandus encapsulates the data handling in its own classes and the
 * user is only burdened with modifying local integrators and the
 * generated mesh. Indeed, a lot of local local integrators are
 * implemented in amandus and they are in subdirectories labeled by
 * the name of the model. See for instance the namespaces
 * LaplaceIntegrators, StokesIntegrators, Elasticity, Advection, or
 * AllenCahn. For files using them, check out @ref Examples.
 *
 * Installation instructions are in the file <a
 * href="https://bitbucket.org/guidokanschat/amandus/overview">README.md</a>
 * at the top level of the Amandus directory hierarchy. deal.II has to
 * be installed before. Since we are developing the latter as well,
 * you will usually need the developer version available on github.
 *
 * The solvers in Amandus are either using <a
 * href="http://faculty.cse.tamu.edu/davis/suitesparse.html">UMFPACK</a>
 * or are based on multilevel overlapping Schwarz methods. Ideally, a
 * program written with the help of Amandus consists only of a short
 * driver file and functions which integrate a residual and a matrix
 * on cells and faces of a mesh.
 *
 * Please refer to the "Modules" tab above for a structured overview
 * of this package.
 */
