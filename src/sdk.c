#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* S is side length and T = sqrt(S) */
#ifndef S
#define S 9
#endif
#ifndef T
#define T 3
#endif

#define STR(A) #A
#define LEN(A) ((int)(sizeof(A)-1))
#define LENSTR(A) LEN(STR(A))

/*
 * b[i][j] is a bitmap showing the permitted values.
 * d[i][j] is 1 if the value of this square is known (i.e. one bit set).
 */
struct board { unsigned int b[S][S]; unsigned int d[S][S]; };

int setval(struct board *b, int i, int j, int v);
int trysolve(struct board *b);

/*
 * Exit and print an error. printflike.
 */
void
die(char *s, ...) {
	va_list l;

	va_start(l, s);
	vfprintf(stderr, s, l);
	va_end(l);
	fprintf(stderr, "\n");
	exit(1);
}

/*
 * Set up a board structure.
 * Initially all values are possible.
 */
void
clearboard(struct board *b) {
	int i, j;

	for (i = 0; i < S; i++)
		for (j = 0; j < S; j++) {
			b->b[i][j] = (1U<<S)-1;
			b->d[i][j] = 0;
		}
}

/*
 * Read a board, drawing simple implecations.
 * We allow 0 and _ as unknown. We also allow based 36.
 * Internally we use 0..8 instead of 1..9.
 */
void
readboard(struct board *b) {
	int i, j, v;
	char c;

	for (i = 0; i < S; i++)
		for (j = 0; j < S; j++) {
			if (scanf(" %c", &c) != 1)
				die("Couldn't read board [%d,%d].", i+1, j+1);
			if (c >= '0' && c <= '9')
				v = c - '0';
			else if (c >= 'a' && c <= 'z')
				v = c - 'a' + 10;
			else if (c >= 'A' && c <= 'Z')
				v = c - 'A' + 10;
			else if (c == '_')
				continue;
			else
				die("Bad entry '%c' [%d,%d].", c, i+1, j+1);
			if (S < 10)
				if (v-- == 0)
					continue;
			if (v < 0 || v >= S)
				die("Out of range '%c' [%d,%d].", c, i+1, j+1);
			if (setval(b, i, j, v) == -1)
					die("Inconsistent setting '%c' [%d,%d]",
					    c, i+1, j+1);
		}
}

/*
 * Print a board, assuming it has been solved.
 */
void
printboard(struct board *b) {
	int i, j, v;

	for (i = 0; i < S; i++) {
		for (j = 0; j < S; j++) {
			if (b->b[i][j] & (b->b[i][j]-1) || !b->d[i][j])
				die("Board not solved!");
			for (v = 0; v < S; v++)
				if (b->b[i][j] == 1U<<v) {
					printf("%*X ", LENSTR(S),
					     S < 10 ? v+1 : v);
					break;
				}
		}
		printf("\n");
	}
	printf("\n");
}

/*
 * Set a value, checking for obvious inconsistencies.
 * Returns -1 if inconsistent.
 * Otherwise returns number of squares influenced by setting this value.
 */
int
setval(struct board *b, int i, int j, int v) {
	int k, l, c;
	struct board ob;

	ob = *b;

	b->d[i][j] = 1;
	b->b[i][j] &= 1U<<v;
	for (k = 0; k < S; k++)
		if (k != i)
			b->b[k][j] &= ~(1U<<v);
	for (l = 0; l < S; l++)
		if (l != j)
			b->b[i][l] &= ~(1U<<v);
	for (k = T*(i/T); k < T*(i/T)+T; k++)
		for (l = T*(j/T); l < T*(j/T)+T; l++)
			if (k != i || j != l)
				b->b[k][l] &= ~(1U<<v);

	c = 0;
	for (k = 0; k < S; k++)
		for (l = 0; l < S; l++) {
			if (b->b[k][l] == 0)
				return -1;
			if (b->b[k][l] != ob.b[k][l])
				c++;
		}

	return c;
}

/*
 * Try all possible remaining values at i,j and see if they lead to
 * a solution, using recursion if necessary.
 */
int
tryval(struct board *b, int i, int j) {
	struct board tb;
	int v, solved;

	if (i < 0 || i >= S || j < 0 || j >= S || !(b->b[i][j] & ((1U<<S)-1)))
		die("tryval called with silly values [%d,%d] %d", i, j,
		    b->b[i][j]);

	solved = 0;
	for (v = 0; v < S; v++) {
		if (b->b[i][j] & 1U<<v) {
			tb = *b;
			if (setval(&tb, i, j, v) == -1)
				die("Asked to setval that doesn't work out");
			if (trysolve(&tb) == 0)
				solved++;
		}
	}

	return -1;
}

/*
 * Walk around the board looking for locations where only one number is
 * possible. Keep doing that until we fill the board or have to guess.
 */
int
trysolve(struct board *b) {
	int i, j, v, c, change, done, bits;
	int i_b, j_b, bits_b;

	change = 1;
	while (change) {
		change = 0;
		done = 0;
		i_b = j_b = -1; /* Most constrained square so far */
		bits_b = S+1;
		for (i = 0; i < S; i++)
			for (j = 0; j < S; j++) {
				bits = 0;
				for (v = 0; v < S; v++) {
					/* Only one value possible and not yet set. */
					if (b->b[i][j] == 1U<<v && !b->d[i][j]) {
						c = setval(b, i, j, v);
						if (c == -1)
							return -1;
						change += c;
					}
					if (b->b[i][j] & 1U<<v)
						bits++;
				}
				if (b->d[i][j])
					done++;
				/* Remember most constrained. */
				if (bits > 1 && bits < bits_b) {
					i_b = i;
					j_b = j;
					bits_b = bits;
				}
			}

		if (done == S*S) {
			printboard(b);
			return 0;
		}
	}

	if (tryval(b, i_b, j_b) != -1)
		return 0;

	return -1;
}

int
main(int argc, char **argv) {
	struct board board, *b = &board;

	if (8*sizeof(b->b)/sizeof(char) < S)
		die("Size of bitmap is %z, too small for values %d", sizeof(b->b), S);

	clearboard(b);

	readboard(b);

	trysolve(b);

	return 0;
}
