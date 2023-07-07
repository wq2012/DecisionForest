/**
 * Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
 * Signal Analysis and Machine Perception Laboratory,
 * Department of Electrical, Computer, and Systems Engineering,
 * Rensselaer Polytechnic Institute, Troy, NY 12180, USA
 */

/**
 * This is the C++ implementation of the hash table data structure.
 * Implemented functionalities:
 *     1. Add an element with its key (the key is an integer).
 *     2. Check if a key exists.
 *     3. Get an element given its key.
 *     4. Get the size of the hash table.
 *     5. Iterate the hash table.
 *
 * This implementation is used by the decision tree package, thus we did
 * implement functions such as remove.
 */

#ifndef HashTable_H
#define HashTable_H

#include <iostream>

/**********************************************
 * Declaration part
 **********************************************/

template <class T>
class HashNode
{
public:
    long key;
    T data;
    HashNode<T> *probe; // for hashing
    HashNode<T> *next;  // for iterating
    HashNode(long key_, T data_);
};

template <class T>
class HashTable
{
private:
    long length;          // number of buckets
    long size_;           // number of elements
    HashNode<T> **table;  // for hashing
    HashNode<T> *head;    // for iterating
    HashNode<T> *tail;    // for iterating
    HashNode<T> *current; // for iterating

public:
    HashTable(long length_);
    ~HashTable();
    long size();
    void add(long key, T data);
    bool has(long key);
    T get(long key);
    void begin(); // start iterating
    bool hasNext();
    HashNode<T> *next();
};

/**********************************************
 * Implementation part
 **********************************************/

template <class T>
HashNode<T>::HashNode(long key_, T data_)
{
    key = key_;
    data = data_;
    probe = NULL;
    next = NULL;
}

template <class T>
HashTable<T>::HashTable(long length_)
{
    length = length_;
    size_ = 0;
    table = new HashNode<T> *[length];
    for (long i = 0; i < length; i++)
    {
        table[i] = NULL;
    }
    head = NULL;
    tail = NULL;
}

template <class T>
HashTable<T>::~HashTable()
{
    HashNode<T> *node = head;
    while (node != NULL)
    {
        HashNode<T> *node2 = node->next;
        delete node;
        node = node2;
    }
    delete[] table;
}

template <class T>
long HashTable<T>::size()
{
    return size_;
}

template <class T>
void HashTable<T>::add(long key, T data)
{
    if (key < 0)
        key = -key;
    long pos = key % length;

    HashNode<T> *current = table[pos];
    while (current != NULL)
    {
        if (current->key == key)
        {
            current->data = data;
            return;
        }
        current = current->probe;
    }

    HashNode<T> *node = new HashNode<T>(key, data);
    if (table[pos] == NULL)
    {
        table[pos] = node;
    }
    else
    {
        node->probe = table[pos];
        table[pos] = node;
    }

    if (head == NULL)
    {
        head = node;
        tail = node;
    }
    else
    {
        tail->next = node;
        tail = tail->next;
    }

    size_++;
}

template <class T>
bool HashTable<T>::has(long key)
{
    if (key < 0)
        key = -key;
    long pos = key % length;
    HashNode<T> *node = table[pos];
    while (node != NULL)
    {
        if (node->key == key)
        {
            return true;
        }
        node = node->probe;
    }
    return false;
}

template <class T>
T HashTable<T>::get(long key)
{
    if (key < 0)
        key = -key;
    long pos = key % length;
    HashNode<T> *node = table[pos];
    while (node != NULL)
    {
        if (node->key == key)
        {
            return node->data;
        }
        node = node->probe;
    }
    std::cout << "Error: element does not exist in HashTable. \n";
    exit(1);
}

template <class T>
void HashTable<T>::begin()
{
    current = head;
}

template <class T>
bool HashTable<T>::hasNext()
{
    return current != NULL;
}

template <class T>
HashNode<T> *HashTable<T>::next()
{
    if (current == NULL)
    {
        std::cout << "Error: next element does not exist in HashTable. \n";
        exit(1);
    }
    HashNode<T> *result = current;
    current = current->next;
    return result;
}

#endif