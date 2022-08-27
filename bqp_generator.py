#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generator of symmetric Binary Quadratic Programming (BQP) instances.

Usage
-----
    python3 bqp_generator.py <instance_size> <minimum_partition_element_size>

Arguments
---------
    instance_size : int
        The size of the instance.
    minimum_partition_element_size : int
        The minimum size of the elements of the partition.

Output
------
    A ".dat" file for each BQP instance created.
"""

import os
import sys

from time import time, ctime
from enum import IntEnum
from typing import List, Union, Any

import numpy as np


# number of generation attempts with random partitions
NUMBER_OF_ATTEMPTS = 1

# properties inherited by all intances
CARDINALITY_SET_IEQ = 1
CARDINALITY_SET_IIN = 0
CARDINALITY_SET_I = 1 + CARDINALITY_SET_IEQ + CARDINALITY_SET_IIN


class BlockType (IntEnum):
    """Class that implements an enumeration of block types.

    Options
    -------
        SYMMETRIC_DD : Symmetric diagonal dominant block.
        GRAM : Gram block.
    """
    SYMMETRIC_DD : int = 1
    GRAM : int = 2


class Block ():
    """Class that represents square matrices.

    Attributes
    ----------
        size : int
            The dimension of the block.
        block_type : BlockType
            The type of the block.
        seed : List[int]
            The list of integers used to generate the block.
        matrix : List[List]
            The list of integers that constitute the block.
        description : string
            A textual description of the block.
        is_symmetric : bool
            A flag indicating whether or not the block is symmetric.

    Raise
    -----
        TypeError : if size is not an int;
        ValueError : if size is negative.
    """

    def __init__ (self, size : int) -> None:
        """
        Parameters
        ----------
            size : int
                The dimension of the block.
        """
        self.size : int = size
        self._block_type : BlockType = self._sort_block_type()
        self._seed : Union[List[int], List[List[int]]] = \
            self._generate_matrix_seed()
        self._matrix : List[List[int]] = self._generate_matrix()

    def __str__ (self) -> str:
        return self.description

    @property
    def size (self) -> int:
        """Get or set the size of the block.

        Update the block's seed and matrix in case of size change.

        Parameters
        ----------
            value : int
                The new block size.

        Raise
        -----
            TypeError : if argument value is not an int;
            ValueError : if argument value is negative.
        """
        return self._size

    @size.setter
    def size (self, value : int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{type(self).__name__}: The block size must be " \
                             f"an int, got a(n) {type(value).__name__}.")
        if value <= 0:
            raise ValueError(f"{type(self).__name__}: The block size must be " \
                             f"positive, got {type(value).__name__}.")
        # when calling the setter self.size(value) during construction time,
        # the object Block has no attribute self._size yet, so the call to
        # getter self.size in the if statement below throws AttributteError;
        # in this case, the exception is caught and self._size properly
        # initialized
        try:
            # if the current size differs from the new one, also update the
            # seed and the matrix AFTER updating the size
            if self.size != value:
                self._size : int = value
                self._seed : Union[List[int], List[List[int]]] = \
                    self._generate_matrix_seed()
                self._matrix : List[List[int]] = self._generate_matrix()
        except AttributeError:
            self._size : int = value

    @property
    def block_type (self) -> BlockType:
        """Get the block's type."""
        return self._block_type

    @property
    def seed (self) -> Union[List[int], List[List[int]]]:
        """Get the block's seed."""
        return self._seed

    @property
    def matrix (self) -> List[List[int]]:
        """Get the block's matrix."""
        return self._matrix

    @property
    def description (self) -> str:
        """Get the textual description of the block.

        Pattern: "<Matrix_Type> = <Matrix>"
        """
        token : str = "?"

        if self.block_type == BlockType.SYMMETRIC_DD:
            token = "DD"
        if self.block_type == BlockType.GRAM:
            token = "Gram"

        return f"{token} = {self.matrix}"

    @property
    def is_symmetric (self) -> bool:
        """Indicate whether the block's matrix is symmetric or not."""
        if self.block_type == BlockType.SYMMETRIC_DD:
            return True
        return False

    def _sort_block_type (self) -> BlockType:
        """Sort a block type from options in BlockType."""
        if np.random.default_rng().integers(0, 2) == 1:
            return BlockType.SYMMETRIC_DD
        return BlockType.GRAM

    def _generate_matrix (self) -> Union[List[List[int]], None]:
        """Generate the block's matrix.

            If the block is DD, the seed is a pair, then:
                matrix := seed[0] + (|block_size|-1)*seed[1] if i=j;
                         -seed[1] if i!=j.

            If the block is Gram, then:
                matrix := seed.T x seed.

            Otherwise:
                matrix := None
        """
        try:
            if self.seed is not None:
                if self.block_type == BlockType.SYMMETRIC_DD:
                    matrix : np.NDArray = np.empty((self.size, self.size))
                    matrix.fill(-self.seed[1])

                    # update the diagonal to make the block DD
                    for i, lst in enumerate(matrix):
                        lst[i] : int = self.seed[0]+(self.size-1)*self.seed[1]

                    return matrix

                if self.block_type == BlockType.GRAM:
                    return np.dot(self.seed.T, self.seed)

            raise ValueError(f"{type(self).__name__}: Seed not available.")

        except ValueError as ve_exc:
            print(f"{type(self).__name__}: matrix generation failed due to " \
                  f"{type(ve_exc).__name__}:")
            print(f" -> {ve_exc}")
            print("Returning None.")
            return None

    def _generate_matrix_seed (self) -> Union[List[int], None]:
        """Generate a seed for the block's matrix generation.

            If the block is DD, a pair (n_1, n_2) of non-negative integers is
            sampled from interval [0, block_size];

            If the block is Gram, a |block_size|x|block_size| matrix of
            integers is sampled from interval [-block_size, block_size].

            Otherwise, seed is None.
        """
        try:
            if self.block_type == BlockType.SYMMETRIC_DD:
                seed_size : int = 2
                return np.random.default_rng().integers(1,
                                                        np.ceil(
                                                            np.log(self.size)
                                                            ),
                                                        seed_size
                                                        )

            if self.block_type == BlockType.GRAM:
                # bound=floor(ln(block_size+10)/(ln(block_size+10)-2))
                x : float = np.log(self.size+10)
                bound : int = np.floor(x/(x-2))
                return np.random.default_rng().integers(-bound,
                                                        bound+1,
                                                        (self.size, self.size)
                                                        )

            raise ValueError(f"{type(self).__name__}: Block type " \
                             f"{self.block_type} does not exist.")

        except ValueError as ve_exc:
            print(f"{type(self).__name__}: matrix seed generation failed " \
                  f"due to {type(ve_exc).__name__}:")
            print(f" -> {ve_exc}")
            return None


class PartitionPattern (IntEnum):
    """Class that implements an enumeration of partition patterns.

    Options
    -------
        REGULAR : all elements of the partition have the same size.
        RANDOM : the elements of the partition have random size.
    """
    REGULAR = 1
    RANDOM = 2


class PartitionElement ():
    """Class that represents elements of partitions.

    Attributes
    ----------
        identifier : int
            The identifier of the element.
        members : List[Any]
            The members of the element.
        size : int
            The size of the element.
        block : Block
            The square matrix associated with the element.
        is_symmetric : bool
            Flag tha indicates whether the element is symmetric or not.
        description : str
            A textual description of the element.

    Raise
    -----
        TypeError if (a) the identifier is not an int, or (b) members is not a
            list.
        ValueError if the identifier is negative.
    """

    def __init__ (self, identifier : int, members : List[Any]):
        """
        Parameters
        ----------
            identifier : int
                The identifier of the element.
            members : List[Any]
                The members of the element.
        """
        self.identifier : int = identifier
        self.members : List[Any] = members
        self._block : Block = Block(self.size)

    def __str__ (self):
        return self.description

    @property
    def identifier (self) -> int:
        """Get or set the identifier of the element.

        Parameters
        ----------
            value : int
                The new identifier of the element.

        Raise
        -----
            TypeError : if the argument value is not an int;
            ValueError : if the argument value is negative.
        """
        return self._identifier

    @identifier.setter
    def identifier (self, value : int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{type(self).__name__}: The identifier of the " \
                             "element must be an int, got a(n) " \
                             f"{type(value).__name__}.")
        if value < 0:
            raise ValueError(f"{type(self).__name__}: The identifier of the " \
                             f"element must be non-negative, got {value}.")
        self._identifier = value

    @property
    def members (self) -> List[Any]:
        """Get or set the members of the element.

        Update the element's block when the size of members change.

        Parameters
        ----------
            value : List[Any]
                The new set of members.

        Raise
        ------
            TypeError if argument value is not a list.
        """
        return self._members

    @members.setter
    def members (self, value : List[Any]) -> None:
        if not isinstance(value, List):
            raise TypeError(f"{type(self).__name__}: Members must be a " \
                             f"list, got a(n) {type(value).__name__}.")
        # when calling the setter self.members(value) during construction time,
        # the object PartitionElement has no attribute self._members yet, so
        # the call to getter self.size below (to save the current size) throws
        # AttributeError; in this case, the exception is caught and
        # self._members properly initialized.
        try:
            # save current size first because self.size may change as soon as the
            # list members is updated
            current_size = self.size
            self._members : List[Any] = value
            # if there was an actual change in the size of the list members, update
            # the block associated with the element
            if current_size != len(value):
                self._block : Block = Block(self.size)
        except AttributeError:
            self._members : List[Any] = value

    @property
    def size (self) -> int:
        """Get the size of the element."""
        return len(self.members)

    @property
    def block (self) -> Block:
        """Get the block associated with the element."""
        return self._block

    @property
    def is_symmetric (self) -> bool:
        """True if the element's block is symmetric and False otherwise."""
        return self.block.is_symmetric

    @property
    def description (self) -> str:
        """Get the element's description."""
        return f"element_{self.identifier} = {self.members}"


class Partition ():
    """Class that implements set partitions.

    Attributes
    ----------
        base_set : List[Any]
            Set over which the partition is built.
        pattern : PartitionPattern
            The pattern of the partition.
        element_size : int
            The size of the elements for regular partitions or the minimum size
            of an element for random partitions.
        elements : List[PartitionElement]
            List of elements (or members) of the partition.

    Raise
    ------
        TypeError : if (a) base set is not a list, (b) pattern is not a
            PartitionPattern and (c) element size is not an int;
        ValueError : if (a) pattern is not a valid option and (b) element size
            is neither greater than 1, nor divides the cardinality of the base
            set (for regular partitions).
    """

    def __init__ (self,
                  base_set : List[Any],
                  pattern : PartitionPattern,
                  element_size : int = 2
                  ) -> None:
        """
        Parameters
        ----------
            base_set : List[Any]
                Set over which the partition is built.
            pattern : Pattern
                The pattern of the partition.
            element_size : int
                The size of the elements for regular partitions or the minimum
                size of an element for random partitions.
        """
        self.base_set : List[Any] = base_set
        self._pattern : PartitionPattern = self._validate_pattern(pattern)
        self._element_size : int = self._validate_element_size(element_size)
        self._elements : List[PartitionElement] = self._create_partition()

    @property
    def base_set (self) -> List[Any]:
        """Get or set the base set of the partition.

        Parameters
        ----------
            value : List[Any]
                The new base set.

        Raise
        ------
            TypeError if argument value is not a list.
        """
        return self._base_set

    @base_set.setter
    def base_set (self, value : List[Any]) -> None:
        if not isinstance(value, list):
            raise TypeError(f"{type(self).__name__}: The base set must be " \
                             f"a list, got a(n) {type(value).__name__}.")
        self._base_set : List[Any] = value

    @property
    def pattern (self) -> PartitionPattern:
        """Get the partition's pattern."""
        return self._pattern

    @property
    def element_size (self) -> int:
        """Get the (minimum) size of the partition's elements."""
        return self._element_size

    @property
    def elements (self) -> List[PartitionElement]:
        """Get the list of elements of the partition."""
        return self._elements

    @property
    def symmetric_elements (self) -> List[PartitionElement]:
        """Get the list of symmetric elements of the partition."""
        return [element for element in self.elements if element.is_symmetric]

    def _validate_pattern (self, value : PartitionPattern) -> PartitionPattern:
        """Check whether or not the argument value is a PartitionPattern.

        Parameters
        ----------
            value : PartitionPattern
                The pattern to be verified.

        Raise
        -----
            TypeError : if value is not a PartitionPattern.
            ValueError : if value is not a valid pattern option.
        """
        if not isinstance(value, PartitionPattern):
            raise TypeError(f"{type(self).__name__}: Pattern must be a " \
                             "PartitionPattern, got a(n) " \
                             f"{type(value).__name__}.")
        if value not in [PartitionPattern.RANDOM, PartitionPattern.REGULAR]:
            raise ValueError(f"{type(self).__name__}: Pattern value must be " \
                             "an option in PartitionPattern.")
        return value

    def _validate_element_size (self, value : int) -> int:
        """Check whether or not the element size is valid.

        Parameters
        ----------
            value : int
                The element size to be verified.

        Raise
        -----
            TypeError : if value is not an int;
            ValueError : if value is neither greater than 1, nor divides the
                cardinality of the base set (for regular partitions).
        """
        if not isinstance(value, int):
            raise TypeError(f"{type(self).__name__}: Element size must " \
                             "be an int, got a(n) " \
                             f"{type(value).__name__}.")
        if value < 2:
            raise ValueError(f"{type(self).__name__}: Element size must " \
                             f"be greater than 1, got {value}.")

        if self.pattern == PartitionPattern.REGULAR:
            if len(self.base_set) % value != 0:
                raise ValueError(f"{type(self).__name__}: Cardinality of " \
                                 f"base set ({len(self.base_set)}) not " \
                                 f"divisible by element size ({value}).")
        return value

    def _create_partition (self) -> List[PartitionElement]:
        """Create a partition of the base set."""
        elements : List = []

        if self.pattern == PartitionPattern.REGULAR:
            baseset_size : int = len(self.base_set)
            number_of_elements : int = int(baseset_size/self.element_size)
            for element_id in range(number_of_elements):
                start : int = element_id * self.element_size
                members : List = self.base_set[start:start+self.element_size]
                elements.append(PartitionElement(element_id, members))

        if self.pattern == PartitionPattern.RANDOM:
            element_id : int = 0
            set_to_be_partitioned : List = self.base_set.copy()
            min_element_size : int = self.element_size

            while True:
                # set the max size of the current element
                max_element_size : int = len(set_to_be_partitioned)
                # sort the size of the current element
                element_size : int = np.random.default_rng().integers(
                    min_element_size,
                    max_element_size+1
                )
                # copy the members of the current element from the set to be
                # partitioned
                members : List = set_to_be_partitioned[:element_size]
                # delete the recently extracted members from the set to be
                # partitioned
                del set_to_be_partitioned[:element_size]
                # if there is still entries in the set to be partitioned but
                # they are not enough to create another element satisfying
                # min_element_size, (a) copy the remaining entries to the
                # current list of members and (b) make the set to be
                # partitioned empty to indicate that the procedure is over
                if 0 < len(set_to_be_partitioned) < min_element_size:
                    members : List = members + set_to_be_partitioned
                    set_to_be_partitioned : List = []
                # add the partition element to the list of elements
                elements.append(PartitionElement(element_id, members))
                # if there are no entries left in the set to be partitioned,
                # the procedure is over, break the loop
                if len(set_to_be_partitioned) == 0:
                    break
                # otherwise, increment the element id and repeat
                element_id =+ 1

        return elements


class Instance ():
    """Class that represents symmetric BQP instances.

    Attributes
    ----------
        size : int
            The size of the instance.
        partition : Partition
            A partition of the set of indices of the instance.
        name : str
            The name of the instance.
        set_n : List[int]
            The list of decision variables indices.
        set_i : List[int]
            The list of matrices indices.
        set_ieq : List[int]
            The list of equality contraints indices.
        set_iin : List[int]
            The list of inequality contraints indices.
        param_b : List[int]
            The list of right-hand side parameters.

    Raise
    -----
        TypeError : if size is not an int;
        ValueError : if size is not greater than one.
    """

    def __init__ (self,
                  size : int,
                  partition_pattern : PartitionPattern,
                  partition_element_size : int = 2
                  ) -> None:
        """
        Parameters
        ----------
            size : int
                The size of the instance.
            partition_pattern : PartitionPattern
                The pattern of the partition of the instance's indices.
            partition_element_size : int
                The size of the elements of the partition, default is 2.
        """
        self.size : int = size
        self._partition : Partition = Partition(self.set_n,
                                                partition_pattern,
                                                partition_element_size
                                                )

    def __str__ (self) -> str:
        return self.name

    @property
    def size (self) -> int:
        """Get or set the size of the instance.

        The size is updated only if the new value differs from the incumbent
        one; if that is the case, the system attempts to generate a new
        partition in order to reflect the modification.

        Parameters
        ----------
            value : int
                The size of the instance.

        Raise
        -----
            TypeError : if value is not an int;
            ValueError : if value not greater than one.
        """
        return self._size

    @size.setter
    def size (self, value : int) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{type(self).__name__}: Instance size must be " \
                            f"an int, got a(n) {type(value).__name__}.")
        if value < 2:
            raise ValueError(f"{type(self).__name__}: Instance size must be " \
                             f"greater than 1, got {value}.")
        # when calling the setter self.size(value) during construction time,
        # the object Instance has no attribute self._size yet, so the call to
        # getter self.size in the if statement below throws AttributteError;
        # the exception is caught and self._size properly initialized
        try:
            if self.size != value:
                self._size : int = value
                self._partition : Partition = Partition(
                    self.set_n,
                    self.partition.pattern,
                    self.partition.element_size
                )
        except AttributeError:
            self._size : int = value

    @property
    def partition (self) -> Partition:
        """Get the instance's partition."""
        return self._partition

    @property
    def set_n (self) -> List[int]:
        """Get the set N of variables indices."""
        return list(range(1, self.size+1))

    @property
    def set_i (self) -> List[int]:
        """Get the set I of matrices indices."""
        return list(range(CARDINALITY_SET_I))

    @property
    def set_ieq (self) -> List[int]:
        """Get the set Ieq of indices of equality constraints."""
        return list(range(1, CARDINALITY_SET_IEQ+1))

    @property
    def set_iin (self) -> List[int]:
        """Get the set Iin of indices of inequality constraints."""
        return list(range(CARDINALITY_SET_IEQ+1, CARDINALITY_SET_I))

    @property
    def param_b (self) -> List[int]:
        """Get the param b (a.k.a. right-hand side) of the constraints."""
        return [int(np.ceil(self.size/2))]

    @property
    def name (self) -> str:
        """Get the name of the instance.

        Pattern is bqp_<n>_<o>x<s>, where:
            (a) <n> = instance's size;
            (b) <o> = number of orbits;
            (c) <s> = orbits' size (R for random sizes, ? for indefinite).
        """
        token : str = "?"
        number_of_orbits : int = len(self.partition.symmetric_elements)

        if number_of_orbits == 1:
            token : int = self.partition.symmetric_elements[0].size
            return f"bqp_{self.size}_{number_of_orbits}x{token}"
        if number_of_orbits > 1:
            if self.partition.pattern == PartitionPattern.REGULAR:
                token = self.partition.element_size
            if self.partition.pattern == PartitionPattern.RANDOM:
                token = "R"
            return f"bqp_{self.size}_{number_of_orbits}x{token}"
        # no symmetries
        return f"bqp_{self.size}"


    def write_to_file (self) -> None:
        """Write the instance's data to a .dat file."""
        try:
            with open(self.name + ".dat", "w", encoding="utf-8") as file:
                file.write(f"set N := {' '.join(str(i) for i in self.set_n)};\n\n")
                file.write(f"set I := {' '.join(str(i) for i in self.set_i)};\n\n")
                file.write(f"set Ieq := {' '.join(str(i) for i in self.set_ieq)};")
                file.write("\n\nparam A :=\n")
                # matrix A_m for m=0, i.e. A_0
                idx_m : int = 0
                for element in self.partition.elements:
                    for i, line in enumerate(element.block.matrix):
                        for j, a_ij in enumerate(line):
                            file.write(f"{idx_m} {element.members[i]} " \
                                       f"{element.members[j]} {a_ij}\n")
                # matrices A_m for m in Ieq
                for idx_m in self.set_ieq:
                    for i in range(self.size):
                        file.write(f"{idx_m} {i+1} {i+1} 1\n")
                file.write(";\n\n")
                file.write(f"param b := 1 {self.param_b[0]} ;\n")
        except OSError as os_exc:
            print(f"{type(self).__name__}: dat file generation failed due " \
                  f"to {type(os_exc).__name__}:")
            print(f" -> {os_exc}")


class InstanceFactory ():
    """Class that implements a factory of symmetric BQP instances.

    Attributes
    ----------
        instance_size : int
            The size of the instance.
        minimum_partition_element_size : int
            The minimum size of the elements of the instance's partition.

    Raise
    -----
        TypeError : if either instance_size or minimum_partition_element_size
            are not int;
        ValueError : if the attributes do not satisfy the following criteria:
            (a) instance_size > 1;
            (b) minimum_partition_element_size > 1;
            (c) instance_size >= minimum_partition_element_size.
   """

    def __init__ (self,
                  instance_size: int,
                  minimum_partition_element_size: int
                  ) -> None:
        """
        Parameters
        ----------
            instance_size : int
                The size of the instance.
            minimum_partition_element_size : int
                The minimum size of the elements of the partition.
        """
        self.instance_size : int = instance_size
        self.minimum_partition_element_size : int = \
            minimum_partition_element_size

    @property
    def instance_size (self) -> int:
        """Get or set the instance size.

        Parameters
        ----------
            value : int
                The size of the instance.

        Raise
        -----
            TypeError : if value is not an int;
            ValueError : if value is not greater than one.
        """
        return self._instance_size

    @instance_size.setter
    def instance_size (self, value : int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{type(self).__name__}: Instance size must be " \
                             f"an int, got a(n) {type(value).__name__}.")
        if value < 2:
            raise ValueError(f"{type(self).__name__}: Instance size must be " \
                             f"greater than 1, got {value}.")
        self._instance_size : int = value

    @property
    def minimum_partition_element_size (self) -> int:
        """Get or set the minimum size of the partition's elements.

        Parameters
        ----------
            value : int
                The minimum size of the partition element.

        Raise
        -----
            TypeError : if value is not an int;
            ValueError : if value (a) is not greater than 1 (b) nor smaller
            than or equal to the instance size.
        """
        return self._minimum_partition_element_size

    @minimum_partition_element_size.setter
    def minimum_partition_element_size (self, value : int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{type(self).__name__}: Minimum partition " \
                             "element size must be an int, got a(n) " \
                             f"{type(value).__name__}.")
        if value < 2:
            raise ValueError(f"{type(self).__name__}: Minimum partition " \
                             "element size must be an greater than 1, got " \
                             f"{value}.")
        if self.instance_size < value:
            raise ValueError(f"{type(self).__name__}: Minimum partition " \
                             f"element size ({value}) cannot exceed " \
                             f"the instance size ({self.instance_size}).")

        self._minimum_partition_element_size : int = value

    def bunch_with_regular_partition (self, write_to_file : bool) -> None:
        """Create a bunch of symmetric BQP instances with regular partitions.

        The size (or cardinality) of the elements of the partition is invariant
        for an instance; it varies however from instance to instance.

        Parameters
        ----------
            write_to_file : bool
                Indicates whether the .dat file must be generated or not.

        Raise
        -----
            TypeError if write_to_file is not a bool.
        """
        if not isinstance(write_to_file, bool):
            raise TypeError(f"{type(self).__name__}: Argument write_to_file " \
                            "must be a bool, got a(n) " \
                            f"{type(write_to_file).__name__}.")

        for element_size in range(self.minimum_partition_element_size,
                                  self.instance_size + 1
                                  ):
            print(f"Atempting regular partition with element size: {element_size}.")
            try:
                instance : Instance = Instance(self.instance_size,
                                               PartitionPattern.REGULAR,
                                               element_size
                                               )
                token : str = "object"
                if write_to_file:
                    instance.write_to_file()
                    token : str = "object and file"
                print(f"Instance {token} generated successfully.\n")

            except ValueError as ve_exc:
                print(f"{type(self).__name__}: instance generation failed " \
                      f"due to {type(ve_exc).__name__}:")
                print(f" -> {ve_exc}")
                print("Skipping to next instance.\n")

    def bunch_with_random_partition (self, write_to_file : bool) -> None:
        """Create a bunch of symmetric BQP instances with random partitions.

        The size (or cardinality) of the elements of the partition varies.

        Parameters
        ----------
            write_to_file : bool
                Indicates whether the .dat file must be generated or not.

        Raise
        -----
            TypeError if write_to_file is not a bool.
        """
        if not isinstance(write_to_file, bool):
            raise TypeError(f"{type(self).__name__}: Argument write_to_file " \
                            "must be a bool, got a(n) " \
                            f"{type(write_to_file).__name__}.")

        counter : int = 0
        while True:
            print(f"Atempting with random partition: " \
                  f"{counter+1}/{NUMBER_OF_ATTEMPTS}.")
            instance : Instance = Instance(self.instance_size,
                                           PartitionPattern.RANDOM,
                                           self.minimum_partition_element_size
                                           )
            token : str = "object"
            if write_to_file:
                instance.write_to_file()
                token : str = "object and file"
            print(f"Instance {token} generated successfully.\n")

            counter += 1
            if counter == NUMBER_OF_ATTEMPTS:
                break

    def run (self) -> None:
        """Run factory to generate all possible symmetric instances.

        The method calls sequentially self.bunch_with_regular_partition() and
        self.bunch_with_regular_partition() with argument write_to_file=True.
        """
        try:
            self.bunch_with_regular_partition(True)
            self.bunch_with_random_partition(True)
        except (TypeError, ValueError) as tve_exc:
            print(f"{type(self).__name__}: factory run failed due to " \
                  f"{type(tve_exc).__name__}:")
            print(f" -> {tve_exc}\n")


def main ():
    """Run the script in full."""
    print(f"{ctime()}: Starting execution of {os.path.basename(__file__)}.\n")
    start = time()

    if len(sys.argv) == 3:
        try:
            factory = InstanceFactory(int(sys.argv[1]), int(sys.argv[2]))
        except (TypeError, ValueError) as tve_exc:
            print(f"Main: factory creation failed due to " \
                  f"{type(tve_exc).__name__}:")
            print(f" -> {tve_exc}\n")
        else:
            factory.run()
    else:
        print("Usage: python3 bqp_generator.py <instance_size> <minimum_" \
              "partition_element_size>\n")

    end = time()
    print(f"{ctime()}: Terminated; time elapsed: {end-start:.2f} seconds.")


if __name__ == "__main__":
    main()
