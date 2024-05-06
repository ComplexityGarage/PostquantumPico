from random import randint
import os

def randbytes(n):
    return os.urandom(n)

# This is not a real shake256. It is an implementation of a simplified hash function. In the future, it will be changed to shake256
class shake_256:
    def __init__(self, input_bytes) -> None:
        self.input_bytes = input_bytes
    def digest(self, hash_length):
        # DJB2 basic hash value
        hash_value = 5381
        for byte in self.input_bytes:
            hash_value = ((hash_value << 5) + hash_value) + byte
            hash_value &= 0xFFFFFFFF  # Keep the hash as a 32-bit value
        
        # Start generating a longer hash
        extended_hash = bytearray()
        
        # Use the hash_value as the seed for a simple PRNG
        seed = hash_value
        while len(extended_hash) < hash_length:
            # Simple PRNG: (a*x + c) % m
            seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
            extended_hash += seed.to_bytes(4, 'little')
        
        return bytes(extended_hash[:hash_length])

class F16():
    """A class used to represent element of field F16
    
    ...
    
    Attributes
    ----------
    a : int
        Decimal number representing element
    
    Methods
    -------
    """
    def __init__(self, a: int):
        if not isinstance(a,int):
            raise TypeError('It must be int')
        else:
            if a>=0 and a<16:
                self.a: int = a
            else:
                raise ValueError(f'4 bit int must be in range 0-15, {a} is not')
    
    def hex(self):
        return format(self.a,'01x')

    def __add__(self,b):
        if isinstance(b,F16):
            return F16(self.a^b.a)
        else:
            return F16(self.a^b)
        
    def __radd__(self,b):
        return self.__add__(b)

    def __sub__(self,b):
        return self.__add__(b)

    def __rsub__(self,b):
        return self.__sub__(b)
    
    def __neg__(self):
        return self
    
    def __mul__(self,b):
        if isinstance(b,int):
            b = F16(b)
        r8 = (self.a&1)*b.a
        r8 ^= (self.a&2)*b.a
        r8 ^= (self.a&4)*b.a
        r8 ^= (self.a&8)*b.a

        #reduction modulo
        r4 = r8 ^ (((r8>>4)&5)*3) #x^4 = x +1, x^6 = x^3 + x^2
        r4 ^= (((r8>>5)&1)*6) #x^5 = x^2 + x
        return F16(r4&0xf)
    
    def __rmul__(self,b):
        return self.__mul__(b)
    
    def _square(self):
        r4 = self.a&1 #1->1^2=1
        r4 ^= (self.a<<1)&4 #x->x^2
        r4 ^= ((self.a>>2)&1)*3#x^2->x^4=x+1
        r4 ^= ((self.a>>3)&1)*12#x^3->x^6=x^3+x^2
        return F16(r4)
    
    def _inv(self):
        # Fermat inversion, a^(n-2)=a^(-1)
        a2 = self._square()
        a4 = a2._square()
        a8 = a4._square()
        a6 = a4*a2
        return a8*a6

    def __truediv__(self,b):
        return self*b._inv()
    
    def __lt__(self,b):
        return self.a<b.a
    
    def __eq__(self,b):
        if isinstance(b,F16):
            return self.a==b.a
        
        else:
            return self.a==b

    def __le__(self,b):
        return self<b or self==b

    def __repr__(self) -> str:
        return str(self.a)
    
class Matrix():
    def __init__(self, M) -> None:
        self.matrix = M
        self.rows = len(M)
        self.cols = len(M[0])

    def T(self):
        res = zeros(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                res[j,i]=self[i,j]
        return res
    
    def __eq__(self,A):
        if self.rows!=A.rows or self.cols!=A.cols:
            return False
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    if self[i,j]!=A[i,j]:
                        return False
        return True
    
    def __add__(self,A):
        if self.rows!=A.rows or self.cols!=A.cols:
            raise ValueError('Shapes of matrices are not compatible')
        res = zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                res[i,j] = self[i,j] + A[i,j]
        return res
    
    def __neg__(self):
        res = zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                res[i,j] = -self[i,j]
        return res
    
    def __sub__(self,A):
        return self+(-A)
    
    def __mul__(self,A):
        if self.cols!=A.rows:
            raise ValueError('Shapes of matrices are not compatible')
        res = zeros(self.rows, A.cols)
        for i in range(self.rows):
            for k in range(A.cols):
                for j in range(self.cols):
                    res[i,k] += self[i,j]*A[j,k]
        return res
    
    def __repr__(self) -> str:
        res=''
        for i in range(self.rows):
            for j in range(self.cols):
                #res += str(self[i,j])+' '
                res += '{:01x}'.format(self[i,j].a)
            res+='\n'
        return res
    
    def __getitem__(self, index):
        if isinstance(index,tuple):
            return self.matrix[index[0]][index[1]]
        if isinstance(index,int):
            return self.matrix[index]
    
    def __setitem__(self, index, value):
        if isinstance(index,tuple):
            self.matrix[index[0]][index[1]] = value
        if isinstance(index,int):
            self.matrix[index] = value
    

def zeros(rows, cols):
    return Matrix([[0 for i in range(cols)] for j in range(rows)])

def hstack(A,B):
    res = []
    for rowA, rowB in zip(A.matrix,B.matrix):
        res.append(rowA+rowB)
    return Matrix(res)

def vstack(A,B):
    return Matrix(A.matrix+B.matrix)


def solve_system(matrix, vector):
    # Check that the number of equations and unknowns are consistent
    if matrix.rows != vector.rows:
        raise ValueError("Number of equations and variables must be the same")

    # Expand the matrix by a vector of right-hand sides
    augmented_matrix = hstack(matrix,vector)

    rows = matrix.rows
    cols = matrix.cols

    # Gauss elimination
    augmented_matrix = Gauss(augmented_matrix)

    # Backward substitution
    solution = [F16(0)] * cols
    for i in range(rows - 1, -1, -1):
        solution[i] = augmented_matrix[i,-1]
        for j in range(i + 1, cols):
            solution[i] -= augmented_matrix[i,j] * solution[j]

    return Matrix([solution]).T()

def Gauss(matrix):
    rows = matrix.rows
    cols = matrix.cols

    # Gauss elimination
    for i in range(rows):
        # We are looking for the first non-zero element in column i
        pivot = None
        for j in range(i, rows):
            if matrix[j,i] != 0:
                pivot = j
                break

        # If a non-zero element is not found, the system of equations is contradictory or ill-defined
        if pivot is None:
            raise ValueError("System of equations is inconsistent or improperly determined ")

        # Swap rows
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]

        # Skaluj aktualny wiersz, aby uzyskać 1 na przekątnej   
        pivot_element = matrix[i][i]
        matrix[i] = [elem/pivot_element for elem in matrix[i]]

        # Elimination of coefficients in the remaining rows
        for j in range(i + 1, rows):
            factor = matrix[j][i]
            matrix[j] = [elem - factor * matrix[i][k] for k, elem in enumerate(matrix[j])]
    return matrix
        
def zeros_f16(rows, cols):
    res = zeros(rows,cols)
    for i in range(rows):
        for j in range(cols):
            res[i,j] = F16(res[i,j])
    return res

def rand_f16(rows, cols, triu=False):
    res = zeros_f16(rows, cols)
    if triu:
        for i in range(rows):
            for j in range(i,cols):
                res[i,j] = F16(randint(0,15))
    else:
        for i in range(rows):
            for j in range(cols):
                res[i,j] = F16(randint(0,15))
    return res

def eye_f16(n):
    res = zeros_f16(n,n)
    for i in range(n):
        res[i,i]=F16(1)
    return res

def upper(M):
    res = zeros_f16(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(i,M.cols):
            if i==j:
                res[i,j] = M[i,j]
            else:
                res[i,j] = M[i,j] + M[j,i]
    return res

def Encode_vec(v : Matrix) -> bytes:
    """Encodes one-row Matrix of F16 into bytes
    
    Parameters
    ----------
    v : Matrix
        Vector to encode
    
    Returns
    -------
    bytes
        Bytes encoding v
    """
    
    vb = []
    for i in range(0,v.cols,2):
        if i==v.cols-1:
            vb.append((v[0,i].a<<4))
        else:
            vb.append((v[0,i].a<<4)^v[0,i+1].a)
    return bytes(vb)

def Encode_matrix(M : Matrix, T : bool = False) -> bytes:
    """Encodes Matrix of F16 into bytes

    Parameters
    ----------
    M : Matrix
        Matrix to encode
    T : bool, optional
        True if matrix is expected to be triangular (default is False)
    
    Returns
    -------
    bytes
        Bytes encoding M
    """
    vec = []
    if(T):
        for i in range(M.rows):
            vec += M[i,i:]
    else:
        for i in range(M.rows):
            vec += M[i]
    return Encode_vec(Matrix([vec]))

def Decode_vec(v : bytes, l : int) -> Matrix:
    """Decodes bytes into one-row Matrix of F16
    
    Parameters
    ----------
    v : bytes
        Vector encoded in bytes
    l : int
        Expected number of elements in vector

    Returns
    -------
    Matrix
        List of decoded elements
    """
    
    vq = []
    for b in v:
        vq.append(F16(b>>4))
        vq.append(F16(b&15))
    return Matrix([vq[:l]])

def Decode_matrix(M : bytes, rows : int, cols : int, T : bool = False) -> Matrix:
    """Decodes bytes into Matrix of F16

    Parameters
    ----------
    M : bytes
        Matrix encoded in bytes
    rows : int
        Expected number of rows
    cols : int
        Expected number of columns
    T : bool, optional
        True if matrix is expected to be triangular (default is False)
    
    Returns
    -------
    Matrix
        Array of decoded elements
    """
    Mq=[]
    if(T):
        M_vec = Decode_vec(M,rows*(rows+1)//2)[0]
        for i in range(rows):
            Mq.append(i*[F16(0)]+M_vec[:cols])
            M_vec=M_vec[cols:]
            cols-=1
    else:
        M_vec = Decode_vec(M,rows*cols)[0]
        for i in range(rows):
            Mq.append(M_vec[:cols])
            M_vec=M_vec[cols:]
    return Matrix(Mq)

class UOV:
    """A class used to represent instance of UOV scheme
    
    ...
    
    Attributes
    ----------
    m : int
        Number of equations
    n : int
        Number of variables
    o : int
        Number of oil variables
    O_bytes : int
        Size of matrix O in bytes
    P1_bytes : int
        Size of matrix P1 in bytes
    
    Methods
    -------
    CompactKeysGen()
        Generate public and private keys in compact form

    ExpandSKey()
        Expand private and public key from compact form.

    Sign()

    ExpandPKey()

    Verify()
    """

    def __init__(self, m : int, n) -> None:
        """
        Parameters
        ----------
        m : int
            Number of equations
        n : int
            Number of variables
        """

        self.m = m
        self.n = n
        self.O_bytes = self.m*(self.n-self.m)//2
        self.P1_bytes = (self.n-self.m)*(self.n-self.m)//2
        self.P2_bytes = (self.n-self.m)*self.m//2
        self.sk_seed_bytes = 32
        self.pk_seed_bytes = 16
        self.salt_bytes = 16
        self.sk_seed = None
    
    def CompactKeysGen(self, sk_seed=None) -> tuple[bytes, bytes, list[Matrix]]:
        """
        Generate private and public key in compact form.

        Returns
        -------
        sk_seed, pk_seed, P3s : tuple[bytes, bytes, list[Matrix]]
            Tuple of: bytes representing seed of secret key, bytes representing seed of public key and list of Matrix representing P3 matrices.
        """
        if(sk_seed is None):
            sk_seed = randbytes(self.sk_seed_bytes)
        self.sk_seed = sk_seed
        S = shake_256(self.sk_seed).digest(self.O_bytes+self.pk_seed_bytes)
        pk_seed = S[:self.pk_seed_bytes]
        O = S[self.pk_seed_bytes:]
        O = Decode_matrix(O,self.n-self.m,self.m)

        P = shake_256(pk_seed).digest(self.m*self.P1_bytes+self.m*self.P2_bytes)#AES can be used
        P3s = []
        for i in range(self.m):
            print(i)
            P1 = Decode_matrix(P[i*self.P1_bytes:(i+1)*self.P1_bytes],self.n-self.m,self.n-self.m, T=True)
            P2 = Decode_matrix(P[self.m*self.P1_bytes+i*self.P2_bytes:self.m*self.P1_bytes+(i+1)*self.P2_bytes],self.n-self.m,self.m)
            P3 = upper(-O.T()*P1*O-O.T()*P2)
            P3s.append(P3)
        return pk_seed, P3s
    
    def ExpandSKey(self) -> tuple[Matrix, list[Matrix], list[Matrix]]:
        """
        Expand private and public key from compact form.

        Returns
        -------
        O, P1s, Ss : tuple[Matrix, list[Matrix], list[Matrix]]
            Tuple of: Matrix representing matrix O, list of Matrix representing P1 matrices and list of Matrix representing S matrices.
        """

        S = shake_256(self.sk_seed).digest(self.O_bytes+self.pk_seed_bytes)
        pk_seed = S[:self.pk_seed_bytes]
        O = S[self.pk_seed_bytes:]
        O = Decode_matrix(O,self.n-self.m,self.m)

        P = shake_256(pk_seed).digest(self.m*self.P1_bytes+self.m*self.P2_bytes)#AES can be used
        Ss = []
        P1s = []
        for i in range(self.m):
            P1 = Decode_matrix(P[i*self.P1_bytes:(i+1)*self.P1_bytes],self.n-self.m,self.n-self.m, T=True)
            P2 = Decode_matrix(P[self.m*self.P1_bytes+i*self.P2_bytes:self.m*self.P1_bytes+(i+1)*self.P2_bytes],self.n-self.m,self.m)
            S = (P1+P1.T())*O+P2
            P1s.append(P1)
            Ss.append(S)
        return O, P1s, Ss
    
    def Sign(self, M): #pkc+skc
        O, P1s, Ss = self.ExpandSKey()
        salt = randbytes(self.salt_bytes)
        t = shake_256(M+salt).digest(self.m//2)
        t = Decode_vec(t,self.m).T()
        for ctrl in range(255):
            v = shake_256(M+salt+self.sk_seed+bytes([ctrl])).digest((self.n-self.m)//2)
            v = Decode_vec(v,self.n-self.m).T()
            L = v.T()*Ss[0]
            for _ in Ss[1:]:
                L = vstack(L,v.T()*_)
            y = Matrix([[(v.T()*_*v)[0,0]] for _ in P1s])
            try:
                x = solve_system(L,t-y)
                break
            except Exception as ex:
                pass

        s = vstack(v+O*x,x)
        return s, salt
    
    def ExpandPKey(self, pk_seed, P3s):
        P12 = shake_256(pk_seed).digest(self.m*self.P1_bytes+self.m*self.P2_bytes)#zmienieć na AES
        Ps = []
        for i in range(self.m):
            P1 = Decode_matrix(P12[i*self.P1_bytes:(i+1)*self.P1_bytes],self.n-self.m,self.n-self.m, T=True)
            P2 = Decode_matrix(P12[self.m*self.P1_bytes+i*self.P2_bytes:self.m*self.P1_bytes+(i+1)*self.P2_bytes],self.n-self.m,self.m)
            P = vstack(hstack(P1,P2),hstack(zeros_f16(self.m,self.n-self.m),P3s[i]))
            Ps.append(P)

        return Ps
    
    def Verify(self,Ps,M,s,salt):
        t = Decode_vec(shake_256(M+salt).digest(self.m//2),self.m)
        res = Matrix([[]])
        for P in Ps:
            res = hstack(res,s.T()*P*s)
        return res==t
    
    def PublicOutputLenght(self):
        return self.salt_bytes+self.pk_seed_bytes+self.m**2*(self.m+1)//4+self.n//2
    
    def PublicOutputToFile(self, salt, pk_seed, s, P3s):
        res = salt+pk_seed+Encode_vec(s.T())
        for P3 in P3s:
            res+=Encode_matrix(P3,T=True)
        with open('signature','wb') as file:
            file.write(res)

    def PrivateKeyToFile(self):
        with open('priv_key','wb') as file:
            file.write(self.sk_seed)

    def SetPrivateKey(self,file_name):
        with open(file_name,'rb') as file:
            self.sk_seed = file.read()
        
    def PublicInputFromFile(self, file_name):
        with open(file_name,'rb') as file:
            res = file.read()
        salt = res[:self.salt_bytes]
        pk_seed = res[self.salt_bytes:self.salt_bytes+self.pk_seed_bytes]
        s = Decode_vec(res[self.salt_bytes+self.pk_seed_bytes:self.salt_bytes+self.pk_seed_bytes+self.n//2],self.n).T()
        res = res[self.salt_bytes+self.pk_seed_bytes+self.n//2:]
        P3s = []
        for i in range(self.m):
            P3 = Decode_matrix(res[i*self.m*(self.m+1)//4:(i+1)*self.m*(self.m+1)//4],self.m,self.m,T=True)
            P3s.append(P3)
        return salt, pk_seed, s, P3s

    def SignFile(self, file_name):
        with open(file_name, 'rb') as file:
            M = file.read()
        return self.Sign(M)

    def VerifyFile(self, file_name, Ps, s, salt):
        with open(file_name, 'rb') as file:
            M = file.read()
        return self.Verify(Ps,M,s,salt)